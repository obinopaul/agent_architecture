"""
Network Multi-Agent Architecture

A mesh-like configuration where multiple interconnected agents collaborate,
with designated input and output points.
"""

import asyncio
import functools
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from utils import get_model, get_common_tools
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT, PLANNER_PROMPT, INTEGRATION_PROMPT

# Define the state for the network agents workflow
class NetworkAgentState(TypedDict):
    """State for network agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    next_agent: str
    agent_states: Dict[str, Dict[str, Any]]
    visited_agents: List[str]
    task_queue: List[Dict[str, str]]
    is_complete: bool

# Define specialized prompts for each agent
COORDINATOR_PROMPT = """You are a coordination agent that manages the flow of information between specialized agents.
Your job is to analyze the current state of a task, determine which specialized agents should be consulted next,
and ensure that the overall workflow is efficient and leads to a high-quality result.

Based on the current state and needs, you must decide which agent to route to next:
- RESEARCHER: For gathering facts, data, and background information
- PLANNER: For organizing ideas and creating structured approaches
- WRITER: For content creation and text generation
- CRITIC: For evaluation and quality improvement
- FINALIZER: When the task is complete and ready for final packaging

You may also add specific sub-tasks to the task queue for agents to work on."""

# Create proper agent prompts
def create_agent_prompt(system_prompt):
    """Create a proper agent prompt template with all required variables."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

# Node functions
def initialize_state(state: NetworkAgentState) -> Dict:
    """Initialize the state for the network workflow."""
    return {
        "current_agent": "coordinator",
        "next_agent": "",
        "agent_states": {
            "researcher": {"data": [], "status": "not_started"},
            "planner": {"outline": "", "status": "not_started"},
            "writer": {"drafts": [], "status": "not_started"},
            "critic": {"feedback": [], "status": "not_started"}
        },
        "visited_agents": [],
        "task_queue": [],
        "is_complete": False
    }

def coordinator_node(state: NetworkAgentState) -> Dict:
    """Coordinator agent that manages the workflow between other agents."""
    # Get the original query and current state
    original_query = state["messages"][0].content if state["messages"] else "No query provided"
    
    # Format the current state for the coordinator
    agent_states = state["agent_states"]
    visited_agents = state["visited_agents"]
    task_queue = state["task_queue"]
    
    # Create the prompt for the coordinator
    llm = get_model()
    
    coordinator_prompt = PromptTemplate(
        template="""You are a workflow coordinator managing a network of specialized agents.
        
        Original Query: {query}
        
        Current State of Agents:
        - Researcher: {researcher_status}
        - Planner: {planner_status}
        - Writer: {writer_status}
        - Critic: {critic_status}
        
        Previously Visited Agents: {visited_agents}
        
        Current Task Queue: {task_queue}
        
        Based on the current state, determine:
        1. Which specialized agent should be consulted next
        2. What specific task they should perform
        3. Whether the workflow is complete
        
        If the workflow is complete, respond with "FINALIZER" as the next agent.
        Otherwise, choose from: "researcher", "planner", "writer", or "critic".
        
        For the chosen agent, specify a focused task related to the original query.
        
        Format your response as:
        Next Agent: [agent name]
        Task: [specific task for the agent]
        Complete: [yes/no]
        """,
        input_variables=["query", "researcher_status", "planner_status", "writer_status", 
                        "critic_status", "visited_agents", "task_queue"]
    )
    
    # Prepare inputs
    researcher_status = f"Status: {agent_states['researcher']['status']}, Data collected: {len(agent_states['researcher']['data'])} items"
    planner_status = f"Status: {agent_states['planner']['status']}, Outline: {'Created' if agent_states['planner']['outline'] else 'Not created'}"
    writer_status = f"Status: {agent_states['writer']['status']}, Drafts: {len(agent_states['writer']['drafts'])}"
    critic_status = f"Status: {agent_states['critic']['status']}, Feedback items: {len(agent_states['critic']['feedback'])}"
    
    visited_agents_str = ", ".join(visited_agents) if visited_agents else "None"
    task_queue_str = "\n".join([f"- {task['agent']}: {task['task']}" for task in task_queue]) if task_queue else "None"
    
    # Get coordinator's decision
    response = llm.invoke(
        coordinator_prompt.format(
            query=original_query,
            researcher_status=researcher_status,
            planner_status=planner_status,
            writer_status=writer_status,
            critic_status=critic_status,
            visited_agents=visited_agents_str,
            task_queue=task_queue_str
        )
    )
    
    # Parse the response to get next agent and task
    response_text = response.content
    
    # Extract next agent, task, and completion status
    next_agent = "finalizer"  # Default
    task = ""
    is_complete = False
    
    for line in response_text.split("\n"):
        if "Next Agent:" in line:
            next_agent = line.split("Next Agent:")[1].strip().lower()
        elif "Task:" in line:
            task = line.split("Task:")[1].strip()
        elif "Complete:" in line:
            is_complete = line.split("Complete:")[1].strip().lower() == "yes"
    
    # Add to visited agents
    visited_agents = state["visited_agents"] + ["coordinator"]
    
    # Add to task queue if not complete
    task_queue = state["task_queue"]
    if not is_complete and next_agent != "finalizer":
        task_queue.append({"agent": next_agent, "task": task})
    
    return {
        "current_agent": "coordinator",
        "next_agent": next_agent,
        "visited_agents": visited_agents,
        "task_queue": task_queue,
        "is_complete": is_complete or next_agent == "finalizer"
    }

def researcher_node(state: NetworkAgentState) -> Dict:
    """Researcher agent that gathers information and data."""
    # Get the task for the researcher
    task = next((t["task"] for t in state["task_queue"] if t["agent"] == "researcher"), 
               "Research relevant information for the query")
    
    original_query = state["messages"][0].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    # Create proper prompt with required variables
    researcher_prompt = create_agent_prompt(RESEARCHER_PROMPT)
    agent = create_react_agent(llm, tools, prompt = researcher_prompt)
    
    # Format query properly for the agent
    message = HumanMessage(content=f"Task: {task}\nOriginal query: {original_query}")
    
    # Execute the agent with proper input structure
    response = agent.invoke({
        "messages": [message],
        "agent_scratchpad": []  # Empty scratchpad initially
    })
    
    # Update the agent state
    agent_states = state["agent_states"]
    agent_states["researcher"]["data"].append({
        "task": task,
        "result": response["messages"][-1].content
    })
    agent_states["researcher"]["status"] = "completed"
    
    # Update visited agents and remove the task from queue
    visited_agents = state["visited_agents"] + ["researcher"]
    task_queue = [t for t in state["task_queue"] if not (t["agent"] == "researcher" and t["task"] == task)]
    
    return {
        "messages": [AIMessage(content=response["messages"][-1].content, name="researcher")],
        "current_agent": "researcher",
        "next_agent": "coordinator",  # Return to coordinator
        "agent_states": agent_states,
        "visited_agents": visited_agents,
        "task_queue": task_queue
    }

def planner_node(state: NetworkAgentState) -> Dict:
    """Planner agent that creates structured approaches and outlines."""
    # Get the task for the planner
    task = next((t["task"] for t in state["task_queue"] if t["agent"] == "planner"), 
               "Create an outline for addressing the query")
    
    original_query = state["messages"][0].content
    
    # Get any research data if available
    research_data = ""
    if "researcher" in state["agent_states"] and state["agent_states"]["researcher"]["data"]:
        for item in state["agent_states"]["researcher"]["data"]:
            research_data += f"\nResearch task: {item['task']}\nFindings: {item['result']}\n"
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    # Create proper prompt with required variables
    planner_prompt = create_agent_prompt(PLANNER_PROMPT)
    agent = create_react_agent(llm, tools, prompt = planner_prompt)
    
    # Format query properly for the agent
    message = HumanMessage(content=f"Task: {task}\nOriginal query: {original_query}\nResearch Data: {research_data}")
    
    # Execute the agent with proper input structure
    response = agent.invoke({
        "messages": [message],
        "agent_scratchpad": []  # Empty scratchpad initially
    })
    
    # Update the agent state
    agent_states = state["agent_states"]
    agent_states["planner"]["outline"] = response["messages"][-1].content
    agent_states["planner"]["status"] = "completed"
    
    # Update visited agents and remove the task from queue
    visited_agents = state["visited_agents"] + ["planner"]
    task_queue = [t for t in state["task_queue"] if not (t["agent"] == "planner" and t["task"] == task)]
    
    return {
        "messages": [AIMessage(content=response["messages"][-1].content, name="planner")],
        "current_agent": "planner",
        "next_agent": "coordinator",  # Return to coordinator
        "agent_states": agent_states,
        "visited_agents": visited_agents,
        "task_queue": task_queue
    }

def writer_node(state: NetworkAgentState) -> Dict:
    """Writer agent that creates content."""
    # Get the task for the writer
    task = next((t["task"] for t in state["task_queue"] if t["agent"] == "writer"), 
               "Write content based on the research and outline")
    
    original_query = state["messages"][0].content
    
    # Get any research data and outline if available
    research_data = ""
    if "researcher" in state["agent_states"] and state["agent_states"]["researcher"]["data"]:
        for item in state["agent_states"]["researcher"]["data"]:
            research_data += f"\nResearch task: {item['task']}\nFindings: {item['result']}\n"
    
    outline = ""
    if "planner" in state["agent_states"] and state["agent_states"]["planner"]["outline"]:
        outline = f"\nOutline:\n{state['agent_states']['planner']['outline']}"
    
    # Get any previous drafts and feedback
    previous_drafts = ""
    if "writer" in state["agent_states"] and state["agent_states"]["writer"]["drafts"]:
        drafts = state["agent_states"]["writer"]["drafts"]
        previous_drafts = f"\nPrevious draft: {drafts[-1]}"
    
    feedback = ""
    if "critic" in state["agent_states"] and state["agent_states"]["critic"]["feedback"]:
        for item in state["agent_states"]["critic"]["feedback"]:
            feedback += f"\nFeedback: {item}\n"
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    # Create proper prompt with required variables
    writer_prompt = create_agent_prompt(WRITER_PROMPT)
    agent = create_react_agent(llm, tools, prompt = writer_prompt)
    
    # Format query properly for the agent
    message = HumanMessage(content=f"Task: {task}\nOriginal query: {original_query}{research_data}{outline}{previous_drafts}{feedback}")
    
    # Execute the agent with proper input structure
    response = agent.invoke({
        "messages": [message],
        "agent_scratchpad": []  # Empty scratchpad initially
    })
    
    # Update the agent state
    agent_states = state["agent_states"]
    agent_states["writer"]["drafts"].append(response["messages"][-1].content)
    agent_states["writer"]["status"] = "completed"
    
    # Update visited agents and remove the task from queue
    visited_agents = state["visited_agents"] + ["writer"]
    task_queue = [t for t in state["task_queue"] if not (t["agent"] == "writer" and t["task"] == task)]
    
    return {
        "messages": [AIMessage(content=response["messages"][-1].content, name="writer")],
        "current_agent": "writer",
        "next_agent": "coordinator",  # Return to coordinator
        "agent_states": agent_states,
        "visited_agents": visited_agents,
        "task_queue": task_queue
    }

def critic_node(state: NetworkAgentState) -> Dict:
    """Critic agent that evaluates and provides feedback."""
    # Get the task for the critic
    task = next((t["task"] for t in state["task_queue"] if t["agent"] == "critic"), 
               "Evaluate the latest draft and provide feedback")
    
    original_query = state["messages"][0].content
    
    # Get the latest draft
    latest_draft = ""
    if "writer" in state["agent_states"] and state["agent_states"]["writer"]["drafts"]:
        latest_draft = state["agent_states"]["writer"]["drafts"][-1]
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    # Create proper prompt with required variables
    critic_prompt = create_agent_prompt(CRITIC_PROMPT)
    agent = create_react_agent(llm, tools, prompt = critic_prompt)
    
    # Format query properly for the agent
    message = HumanMessage(content=f"Task: {task}\nOriginal query: {original_query}\nLatest draft:\n{latest_draft}")
    
    # Execute the agent with proper input structure
    response = agent.invoke({
        "messages": [message],
        "agent_scratchpad": []  # Empty scratchpad initially
    })
    
    # Update the agent state
    agent_states = state["agent_states"]
    agent_states["critic"]["feedback"].append(response["messages"][-1].content)
    agent_states["critic"]["status"] = "completed"
    
    # Update visited agents and remove the task from queue
    visited_agents = state["visited_agents"] + ["critic"]
    task_queue = [t for t in state["task_queue"] if not (t["agent"] == "critic" and t["task"] == task)]
    
    return {
        "messages": [AIMessage(content=response["messages"][-1].content, name="critic")],
        "current_agent": "critic",
        "next_agent": "coordinator",  # Return to coordinator
        "agent_states": agent_states,
        "visited_agents": visited_agents,
        "task_queue": task_queue
    }

def finalizer_node(state: NetworkAgentState) -> Dict:
    """Finalizer agent that produces the final output."""
    # Get the original query
    original_query = state["messages"][0].content
    
    # Collect information from all agents
    research_data = ""
    if "researcher" in state["agent_states"] and state["agent_states"]["researcher"]["data"]:
        for item in state["agent_states"]["researcher"]["data"]:
            research_data += f"\nResearch task: {item['task']}\nFindings: {item['result']}\n"
    
    outline = ""
    if "planner" in state["agent_states"] and state["agent_states"]["planner"]["outline"]:
        outline = f"\nOutline:\n{state['agent_states']['planner']['outline']}"
    
    latest_draft = ""
    if "writer" in state["agent_states"] and state["agent_states"]["writer"]["drafts"]:
        latest_draft = state["agent_states"]["writer"]["drafts"][-1]
    
    feedback = ""
    if "critic" in state["agent_states"] and state["agent_states"]["critic"]["feedback"]:
        for item in state["agent_states"]["critic"]["feedback"]:
            feedback += f"\nFeedback: {item}\n"
    
    # Create the finalizer
    llm = get_model()
    finalizer_prompt = PromptTemplate(
        template="""You are responsible for finalizing a response after multiple specialized agents have contributed.
        
        Original Query: {query}
        
        Research Data:
        {research_data}
        
        Outline:
        {outline}
        
        Latest Draft:
        {latest_draft}
        
        Feedback from Critic:
        {feedback}
        
        Your task is to create a final, polished response that:
        1. Directly addresses the original query
        2. Incorporates the research, follows the outline structure
        3. Builds on the latest draft
        4. Addresses the feedback from the critic
        
        Provide a comprehensive, well-structured response that represents the culmination of the collaborative process.
        """,
        input_variables=["query", "research_data", "outline", "latest_draft", "feedback"]
    )
    
    # Generate the final response
    response = llm.invoke(
        finalizer_prompt.format(
            query=original_query,
            research_data=research_data,
            outline=outline,
            latest_draft=latest_draft,
            feedback=feedback
        )
    )
    
    # Return the finalized response
    return {
        "messages": [AIMessage(content=response.content, name="finalizer")],
        "is_complete": True
    }

def route_state(state: NetworkAgentState) -> Literal["coordinator", "researcher", "planner", "writer", "critic", "finalizer"]:
    """Route to the next agent based on state."""
    if state["is_complete"]:
        return "finalizer"
    
    next_agent = state["next_agent"]
    if next_agent in ["researcher", "planner", "writer", "critic", "finalizer"]:
        return next_agent
    else:
        return "coordinator"  # Default

def create_network_agent_workflow():
    """Create and return the network agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(NetworkAgentState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalizer", finalizer_node)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "coordinator")
    
    # Add a router node instead of using empty string
    workflow.add_node("router", lambda x: x)
    
    # Add conditional edges from the router node
    workflow.add_conditional_edges(
        "router",
        route_state,
        {
            "coordinator": "coordinator",
            "researcher": "researcher",
            "planner": "planner",
            "writer": "writer",
            "critic": "critic",
            "finalizer": "finalizer"
        }
    )
    
    # Add edges back to router
    workflow.add_edge("coordinator", "router")
    workflow.add_edge("researcher", "router")
    workflow.add_edge("planner", "router")
    workflow.add_edge("writer", "router")
    workflow.add_edge("critic", "router")
    
    # Add final edge
    workflow.add_edge("finalizer", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# Example usage
def run_network_example(query: str):
    """Run an example query through the network agent workflow."""
    workflow = create_network_agent_workflow()
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    config = {"configurable": {"thread_id": "1"}}
    # Run the workflow
    result = workflow.invoke(initial_state, config)
    
    # Get the history of visited agents
    visited_agents = result.get("visited_agents", [])
    
    # Get the final response
    final_response = result["messages"][-1].content
    
    return {
        "visited_agents": visited_agents,
        "final_response": final_response
    }

if __name__ == "__main__":
    # Example query
    query = "Explain the ethical considerations around artificial intelligence development and how they might be addressed in policy."
    
    # Run the example
    result = run_network_example(query)
    
    print("\n--- Network Agent Workflow ---")
    print(f"Agent collaboration path: {' -> '.join(result['visited_agents'])}")
    print("\nFinal Response:")
    print(result["final_response"])