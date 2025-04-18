"""
Sequential Multi-Agent Architecture Using Command Function

In this architecture, agents work in order, passing results from one to the next,
with clear input and output points. This implementation uses the Command approach for flow control.
"""

import functools
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from utils import get_model, get_common_tools
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT, INTEGRATION_PROMPT

def create_agent_prompt(system_prompt):
    """Create a proper agent prompt template with all required variables."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

# Define the state for the sequential agents workflow
class SequentialAgentState(TypedDict):
    """State for sequential agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_phase: str
    intermediate_results: Dict[str, Any]

# Agent node functions using Command
def researcher_node(state: SequentialAgentState) -> Command[Literal["writer"]]:
    """Researcher agent that handles information gathering."""
    # Get the query
    query = state["messages"][-1].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    research_prompt = create_agent_prompt(RESEARCHER_PROMPT)
    agent = create_react_agent(llm, tools, prompt=research_prompt)
    
    # Execute the agent with messages
    response = agent.invoke({
        "messages": [HumanMessage(content=query)],
        "agent_scratchpad": []
    })
    
    # Get the response content
    agent_response = response["messages"][-1].content
    
    # Store the result in intermediate_results
    intermediate_results = state.get("intermediate_results", {})
    intermediate_results["researcher"] = agent_response
    
    # Return Command to go to writer and update state
    return Command(
        goto="writer",
        update={
            "messages": [AIMessage(content=agent_response, name="researcher")],
            "current_phase": "researcher",
            "intermediate_results": intermediate_results
        }
    )

def writer_node(state: SequentialAgentState) -> Command[Literal["critic"]]:
    """Writer agent that handles content creation tasks."""
    # Get the original query and research results
    original_query = state["messages"][0].content if len(state["messages"]) > 1 else ""
    research_results = state["intermediate_results"].get("researcher", "")
    
    # Combine information for the writer
    combined_query = f"Original query: {original_query}\n\nResearch results: {research_results}\n\nPlease write content based on this information."
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    writing_prompt = create_agent_prompt(WRITER_PROMPT)
    agent = create_react_agent(llm, tools, prompt=writing_prompt)
    
    # Execute the agent
    response = agent.invoke({
        "messages": [HumanMessage(content=combined_query)],
        "agent_scratchpad": []
    })
    
    # Get the response content
    agent_response = response["messages"][-1].content
    
    # Update intermediate results
    intermediate_results = state.get("intermediate_results", {})
    intermediate_results["writer"] = agent_response
    
    # Return Command to go to critic and update state
    return Command(
        goto="critic",
        update={
            "messages": [AIMessage(content=agent_response, name="writer")],
            "current_phase": "writer",
            "intermediate_results": intermediate_results
        }
    )

def critic_node(state: SequentialAgentState) -> Command[Literal["integrator"]]:
    """Critic agent that handles evaluation and feedback."""
    # Get original query, research results, and written content
    original_query = state["messages"][0].content if len(state["messages"]) > 1 else ""
    research_results = state["intermediate_results"].get("researcher", "")
    written_content = state["intermediate_results"].get("writer", "")
    
    # Combine information for the critic
    combined_query = f"Original query: {original_query}\n\nResearch results: {research_results}\n\nWritten content: {written_content}\n\nPlease provide critique and feedback on this content."
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    critique_prompt = create_agent_prompt(CRITIC_PROMPT)
    agent = create_react_agent(llm, tools, prompt=critique_prompt)
    
    # Execute the agent
    response = agent.invoke({
        "messages": [HumanMessage(content=combined_query)],
        "agent_scratchpad": []
    })
    
    # Get the response content
    agent_response = response["messages"][-1].content
    
    # Update intermediate results
    intermediate_results = state.get("intermediate_results", {})
    intermediate_results["critic"] = agent_response
    
    # Return Command to go to integrator and update state
    return Command(
        goto="integrator",
        update={
            "messages": [AIMessage(content=agent_response, name="critic")],
            "current_phase": "critic",
            "intermediate_results": intermediate_results
        }
    )

def integrator_node(state: SequentialAgentState) -> Command[Literal[END]]:
    """Integrator agent that combines all previous results."""
    # Get all results
    original_query = state["messages"][0].content if len(state["messages"]) > 1 else ""
    research_results = state["intermediate_results"].get("researcher", "")
    written_content = state["intermediate_results"].get("writer", "")
    critique = state["intermediate_results"].get("critic", "")
    
    # Combine all information for the integrator
    combined_query = f"""
    Original query: {original_query}
    
    Research results: {research_results}
    
    Written content: {written_content}
    
    Critique: {critique}
    
    Please provide a final integrated response that incorporates all of the above.
    """
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    integration_prompt = create_agent_prompt(INTEGRATION_PROMPT)
    agent = create_react_agent(llm, tools, prompt=integration_prompt)
    
    # Execute the agent
    response = agent.invoke({
        "messages": [HumanMessage(content=combined_query)],
        "agent_scratchpad": []
    })
    
    # Get the response content
    agent_response = response["messages"][-1].content
    
    # Update intermediate results
    intermediate_results = state.get("intermediate_results", {})
    intermediate_results["integrator"] = agent_response
    
    # Return Command to go to END and update state
    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=agent_response, name="integrator")],
            "current_phase": "integrator",
            "intermediate_results": intermediate_results
        }
    )

def create_sequential_agent_workflow():
    """Create and return the sequential agent workflow using Command approach."""
    # Create the workflow graph
    workflow = StateGraph(SequentialAgentState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("integrator", integrator_node)
    
    # Add starting edge - the rest of the flow is managed by Command returns
    workflow.add_edge(START, "researcher")
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# Example usage
def run_sequential_example(query: str):
    """Run an example query through the sequential agent workflow."""
    workflow = create_sequential_agent_workflow()
    config = {"configurable": {"thread_id": "1"}} 
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_phase": "",
        "intermediate_results": {}
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state, config)
    
    # Extract the final response
    final_response = result["messages"][-1].content
    intermediate_results = result["intermediate_results"]
    
    return {
        "final_response": final_response,
        "intermediate_results": intermediate_results
    }

if __name__ == "__main__":
    # Example query
    query = "Explain the potential environmental impacts of widespread electric vehicle adoption."
    
    # Run the example
    result = run_sequential_example(query)
    
    print("\nSequential Workflow Results:\n")
    print("--- Research Phase ---")
    print(result["intermediate_results"]["researcher"])
    print("\n--- Writing Phase ---")
    print(result["intermediate_results"]["writer"])
    print("\n--- Critic Phase ---")
    print(result["intermediate_results"]["critic"])
    print("\n--- Final Integrated Response ---")
    print(result["final_response"])