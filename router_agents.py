"""
Router Multi-Agent Architecture

In this architecture, one agent directs input to different output paths based on task requirements.
The router determines which specialized agent is best suited to handle a given query.
"""

import functools
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from utils import get_model, get_common_tools
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT, PLANNER_PROMPT

def create_agent_prompt(system_prompt):
    """Create a proper agent prompt template with all required variables."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
# Define the state for the router agents workflow
class RouterAgentState(TypedDict):
    """State for router agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    category: str
    agent_response: str

# Node functions
def router_node(state: RouterAgentState) -> Dict:
    """Router agent that categorizes queries and routes them to appropriate specialized agents."""
    # Get the query
    query = state["messages"][-1].content
    
    # Create the routing agent
    llm = get_model()
    
    router_prompt = PromptTemplate(
        template="""You are a specialized routing agent that directs incoming queries to the most appropriate specialized agent.
        
        Based on the query, categorize it into EXACTLY ONE of the following categories:
        - RESEARCH: Questions requiring factual information, data gathering, or background research.
        - WRITING: Requests for content creation, summarization, or text generation.
        - PLANNING: Tasks involving organization, scheduling, step-by-step processes, or strategy.
        - CRITIQUE: Requests for evaluation, feedback, or analysis of ideas, text, or plans.
        
        Query: {query}
        
        Return only the category name (e.g., "RESEARCH") with no additional text or explanation.
        """,
        input_variables=["query"]
    )
    
    # Execute the routing
    chain = router_prompt | llm
    category = chain.invoke({"query": query}).content.strip()
    
    # Ensure the category is one of the allowed values
    allowed_categories = ["RESEARCH", "WRITING", "PLANNING", "CRITIQUE"]
    if category not in allowed_categories:
        # Default to RESEARCH if an invalid category is returned
        category = "RESEARCH"
    
    # Return the category
    return {
        "category": category
    }

def research_agent_node(state: RouterAgentState) -> Dict:
    """Research agent that handles information gathering tasks."""
    # Get the query
    query = state["messages"][-1].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    research_prompt = create_agent_prompt(RESEARCHER_PROMPT)
    agent = create_react_agent(llm, tools, prompt = research_prompt)
    
    # Execute the agent
    response = agent.invoke({"messages": query,
                             "agent_scratchpad": [] })
    
    # Return the response
    return {
        "agent_response": response["messages"][-1].content,
        "messages": [AIMessage(content=response["messages"][-1].content, name="research_agent")]
    }

def writing_agent_node(state: RouterAgentState) -> Dict:
    """Writing agent that handles content creation tasks."""
    # Get the query
    query = state["messages"][-1].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    writing_prompt = create_agent_prompt(WRITER_PROMPT)
    agent = create_react_agent(llm, tools, prompt = writing_prompt)
    
    # Execute the agent
    response = agent.invoke({"messages": query,
                             "agent_scratchpad": [] })
    
    # Return the response
    return {
        "agent_response": response["messages"][-1].content,
        "messages": [AIMessage(content=response["messages"][-1].content, name="writing_agent")]
    }

def planning_agent_node(state: RouterAgentState) -> Dict:
    """Planning agent that handles organizational and strategic tasks."""
    # Get the query
    query = state["messages"][-1].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    planning_prompt = create_agent_prompt(PLANNER_PROMPT)
    agent = create_react_agent(llm, tools, prompt = planning_prompt)
    
    # Execute the agent
    response = agent.invoke({"messages": query,
                             "agent_scratchpad": [] })
    
    # Return the response
    return {
        "agent_response": response["messages"][-1].content,
        "messages": [AIMessage(content=response["messages"][-1].content, name="planning_agent")]
    }

def critique_agent_node(state: RouterAgentState) -> Dict:
    """Critique agent that handles evaluation and feedback tasks."""
    # Get the query
    query = state["messages"][-1].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    critique_prompt = create_agent_prompt(CRITIC_PROMPT)
    agent = create_react_agent(llm, tools, prompt = critique_prompt)
    
    # Execute the agent
    response = agent.invoke({"messages": query,
                             "agent_scratchpad": [] })
    
    # Return the response
    return {
        "agent_response": response["messages"][-1].content,
        "messages": [AIMessage(content=response["messages"][-1].content, name="critique_agent")]
    }

def route_query(state: RouterAgentState) -> Literal["research_agent", "writing_agent", "planning_agent", "critique_agent"]:
    """Route the query to the appropriate agent based on category."""
    category = state["category"]
    
    if category == "RESEARCH":
        return "research_agent"
    elif category == "WRITING":
        return "writing_agent"
    elif category == "PLANNING":
        return "planning_agent"
    elif category == "CRITIQUE":
        return "critique_agent"
    else:
        # Default to research agent
        return "research_agent"

def create_router_agent_workflow():
    """Create and return the router agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(RouterAgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("writing_agent", writing_agent_node)
    workflow.add_node("planning_agent", planning_agent_node)
    workflow.add_node("critique_agent", critique_agent_node)
    
    # Add edges
    workflow.add_edge(START, "router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "research_agent": "research_agent",
            "writing_agent": "writing_agent",
            "planning_agent": "planning_agent",
            "critique_agent": "critique_agent"
        }
    )
    
    # Add final edges
    workflow.add_edge("research_agent", END)
    workflow.add_edge("writing_agent", END)
    workflow.add_edge("planning_agent", END)
    workflow.add_edge("critique_agent", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# Example usage
def run_router_example(query: str):
    """Run an example query through the router agent workflow."""
    workflow = create_router_agent_workflow()
    config = {"configurable": {"thread_id": "1"}} 
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state, config)
    
    # Return the results
    return {
        "category": result["category"],
        "agent_response": result["agent_response"],
        "final_message": result["messages"][-1].content
    }

if __name__ == "__main__":
    # Example queries for different categories
    queries = {
        "research": "What are the main factors contributing to climate change?",
        "writing": "Write a marketing email announcing our new product launch.",
        "planning": "Help me create a 30-day fitness plan for beginners.",
        "critique": "Review this paragraph and provide feedback: 'The company's profits increased by 15% last quarter, despite industry challenges.'"
    }
    
    # Run each example
    for category, query in queries.items():
        print(f"\n\n--- Testing {category.upper()} query ---")
        print(f"Query: {query}")
        
        result = run_router_example(query)
        
        print(f"Routed to: {result['category']}")
        print(f"Response: {result['final_message']}")