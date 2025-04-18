"""
Loop Multi-Agent Architecture

Similar to sequential but includes a feedback mechanism where output can be cycled back as input.
This allows for iterative improvement or refinement of results.
"""

"""
Loop Multi-Agent Architecture

Similar to sequential but includes a feedback mechanism where output can be cycled back as input.
This allows for iterative improvement or refinement of results.
"""

import functools
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from utils import get_model, get_common_tools, create_agent_prompt
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT

# Define the state for the loop agents workflow
class LoopAgentState(TypedDict):
    """State for loop agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_draft: str
    feedback: List[str]
    iterations: int
    max_iterations: int
    is_complete: bool

# Node functions
def initialize_state(state: LoopAgentState) -> Dict:
    """Initialize the state for the workflow."""
    return {
        "current_draft": "",
        "feedback": [],
        "iterations": 0,
        "max_iterations": 3,  # Default maximum number of iterations
        "is_complete": False
    }

def research_node(state: LoopAgentState) -> Dict:
    """Research agent node that gathers information."""
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    # Get the original query from messages
    original_query = state["messages"][0].content
    
    # Create a specialized prompt for research
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCHER_PROMPT),

    ])
    
    # Compile previous work and feedback
    previous_work = ""
    if state["current_draft"]:
        previous_work += f"\nCurrent Draft:\n{state['current_draft']}\n"
    
    if state["feedback"]:
        previous_work += "\nFeedback:\n"
        for i, feedback in enumerate(state["feedback"]):
            previous_work += f"  {i+1}. {feedback}\n"
    
    # Partial the prompt with topic and previous work
    research_prompt = research_prompt.partial(
        topic=original_query,
        previous_work=previous_work
    )
    
    # Create the agent with tools
    agent = create_react_agent(llm, tools, prompt=research_prompt)
    
    query = """Please research the following topic:
        {topic}
        Previous drafts and feedback (if any):
        {previous_work}
        Focus on finding key facts, data, and perspectives that will help create a comprehensive response.
        """
        
    query = query.format(
        topic=original_query,
        previous_work=previous_work
    )
    
    response = agent.invoke({
        'messages': query
    })
    
    
    # Add the research to the state
    return {
        "messages": [AIMessage(content=response["messages"][-1].content, name="researcher")],
        "iterations": state["iterations"] + 1
    }
    

def draft_node(state: LoopAgentState) -> Dict:
    """Writer agent node that creates or refines the draft."""
    # Get the research from the last message
    research = state["messages"][-1].content
    original_query = state["messages"][0].content
    
    # Create the agent
    llm = get_model()
    writer_prompt = PromptTemplate(
        template="""You are a skilled writer tasked with creating a well-structured, informative response.
        
        Original request: {original_query}
        
        Research provided: {research}
        
        Previous drafts and feedback (if any):
        {previous_work}
        
        Your task is to {task}. Focus on clarity, organization, and accuracy. 
        The response should directly address the original request and incorporate the research provided.
        """,
        input_variables=["original_query", "research", "previous_work", "task"]
    )
    
    # Determine if this is a new draft or revision
    task = "create an initial draft" if not state["current_draft"] else "revise the current draft based on feedback"
    
    # Compile previous work and feedback
    previous_work = ""
    if state["current_draft"]:
        previous_work += f"\nCurrent Draft:\n{state['current_draft']}\n"
    
    if state["feedback"]:
        previous_work += "\nFeedback:\n"
        for i, feedback in enumerate(state["feedback"]):
            previous_work += f"  {i+1}. {feedback}\n"
    
    # Execute the chain
    chain = writer_prompt | llm
    response = chain.invoke({
        "original_query": original_query,
        "research": research,
        "previous_work": previous_work,
        "task": task
    })
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=response.content, name="writer")],
        "current_draft": response.content
    }

def evaluate_node(state: LoopAgentState) -> Dict:
    """Critic agent node that evaluates the current draft and provides feedback."""
    # Get the current draft
    current_draft = state["current_draft"]
    original_query = state["messages"][0].content
    
    # Create the agent
    llm = get_model()
    critic_prompt = PromptTemplate(
        template="""You are a critical evaluator tasked with providing constructive feedback on the following draft.
        
        Original request: {original_query}
        
        Current draft:
        {current_draft}
        
        Previous feedback (if any):
        {previous_feedback}
        
        Iteration {iteration} of {max_iterations}.
        
        Evaluate the draft on:
        1. Relevance to the original request
        2. Accuracy and factual correctness
        3. Clarity and organization
        4. Completeness
        5. Overall quality
        
        Provide constructive feedback that can be used to improve the draft.
        At the end, include an overall evaluation score from 1-10.
        
        If the draft scores 8 or higher, or if this is the final iteration, indicate "COMPLETE".
        Otherwise, indicate "NEEDS REVISION".
        """,
        input_variables=["original_query", "current_draft", "previous_feedback", "iteration", "max_iterations"]
    )
    
    # Format previous feedback
    previous_feedback = ""
    if state["feedback"]:
        for i, feedback in enumerate(state["feedback"]):
            previous_feedback += f"  {i+1}. {feedback}\n"
    
    # Execute the chain
    chain = critic_prompt | llm
    response = chain.invoke({
        "original_query": original_query,
        "current_draft": current_draft,
        "previous_feedback": previous_feedback,
        "iteration": state["iterations"],
        "max_iterations": state["max_iterations"]
    })
    
    # Determine if complete
    is_complete = (
        "COMPLETE" in response.content or 
        state["iterations"] >= state["max_iterations"]
    )
    
    # Add feedback to the list
    feedback = state["feedback"] + [response.content]
    
    # Return the updated state
    return {
        "messages": [AIMessage(content=response.content, name="critic")],
        "feedback": feedback,
        "is_complete": is_complete
    }

def finalize_node(state: LoopAgentState) -> Dict:
    """Finalize the response."""
    # Get the final draft
    final_draft = state["current_draft"]
    
    # Create a final summary message
    llm = get_model()
    summary_prompt = PromptTemplate(
        template="""You are tasked with finalizing a response after {iterations} iterations of feedback and revision.
        
        Original request: {original_query}
        
        Final draft:
        {final_draft}
        
        Create a brief summary of this response process, including:
        1. How the response evolved through iterations
        2. Key improvements made
        3. A note that this is the final version after iterative refinement
        
        Then present the final response.
        """,
        input_variables=["iterations", "original_query", "final_draft"]
    )
    
    # Execute the chain
    chain = summary_prompt | llm
    response = chain.invoke({
        "iterations": state["iterations"],
        "original_query": state["messages"][0].content,
        "final_draft": final_draft
    })
    
    # Return the final message
    return {
        "messages": [AIMessage(content=response.content, name="finalizer")]
    }

def should_continue(state: LoopAgentState) -> Literal["continue", "finalize"]:
    """Determine if the loop should continue or finish."""
    if state["is_complete"]:
        return "finalize"
    else:
        return "continue"

def create_loop_agent_workflow():
    """Create and return the loop agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(LoopAgentState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("research", research_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "research")
    workflow.add_edge("research", "draft")
    workflow.add_edge("draft", "evaluate")
    
    # Add conditional edges for the loop
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "continue": "research",  # Loop back to research
            "finalize": "finalize"   # Move to finalization
        }
    )
    
    workflow.add_edge("finalize", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# Example usage
def run_loop_example(query: str, max_iterations: int = 3):
    """Run an example query through the loop agent workflow."""
    workflow = create_loop_agent_workflow()

    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": max_iterations
    }
    config = {"configurable": {"thread_id": "1"}} 
    
    # Run the workflow
    result = workflow.invoke(initial_state, config)
    
    # Process the results
    iterations = result["iterations"]
    feedback_history = result["feedback"]
    final_draft = result["current_draft"]
    final_message = result["messages"][-1].content
    
    return {
        "iterations": iterations,
        "feedback_history": feedback_history,
        "final_draft": final_draft,
        "final_message": final_message
    }

if __name__ == "__main__":
    # Example query
    query = "Explain the concept of quantum computing and its potential applications for a general audience."
    
    # Run the example
    result = run_loop_example(query, max_iterations=2)
    
    print(f"\nLoop Workflow completed in {result['iterations']} iterations\n")
    
    print("--- Feedback History ---")
    for i, feedback in enumerate(result["feedback_history"]):
        print(f"\nFeedback {i+1}:")
        print(feedback)
    
    print("\n--- Final Message ---")
    print(result["final_message"])