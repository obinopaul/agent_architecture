"""
Parallel Multi-Agent Architecture

In this architecture, multiple agents work simultaneously on tasks, with designated input and output points.
"""

import asyncio
from typing import List, Dict, Any, Annotated, Sequence, TypedDict
import operator
import functools

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from utils import get_model, get_common_tools, create_agent_prompt
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT

# Define the state for the parallel agents' workflow
class ParallelAgentState(TypedDict):
    """State for parallel agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sub_tasks: List[Dict[str, str]]
    current_task: Dict[str, str]
    results: Dict[str, Any]

# Node functions
async def split_task(state: ParallelAgentState) -> Dict[str, List[Dict[str, str]]]:
    """Split the input task into multiple parallel sub-tasks."""
    llm = get_model()
    
    # Template for splitting the task
    prompt = PromptTemplate(
        template="""You are a task decomposition specialist. 
        Given a complex query, break it down into 3 specific sub-tasks that can be executed in parallel.
        Each sub-task should be independent and focused on a different aspect of the query.
        
        Format your response as a comma-separated list of sub-tasks.
        
        Query: {query}
        """,
        input_variables=["query"]
    )
    
    # Process the query to generate sub-tasks
    chain = prompt | llm
    query = state["messages"][-1].content
    response = await chain.ainvoke({"query": query})
    
    # Parse the response into a list of sub-tasks
    sub_tasks = [task.strip() for task in response.content.split(',')]
    sub_task_dicts = [{"task": task} for task in sub_tasks]
    
    return {
        "sub_tasks": sub_task_dicts
    }

async def run_agent(state: ParallelAgentState, agent_name: str, agent_prompt: str) -> Dict[str, Any]:
    """Run a specific agent on a task."""
    current_task = state["current_task"]["task"]
    
    # Get tools first before referencing them
    tools = get_common_tools()
    
    # Create a prompt for the agent
    agent_prompt_template = create_agent_prompt(agent_prompt)

    agent_prompt_template = ChatPromptTemplate.from_messages([
         SystemMessage(content="""{agent_prompt}
        
        AVAILABLE TOOLS:
        {tools}
        
        TOOL USAGE PROTOCOL:
        - You have access to the following tools: [{tool_names}]
        - BEFORE using any tool, EXPLICITLY state:
        1. WHY you are using this tool
        2. WHAT specific information you hope to retrieve
        3. HOW this information will help solve the task
        
        """),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Partial the prompt with tools and tool names
    agent_prompt_template = agent_prompt_template.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
        agent_prompt=agent_prompt
    )
    
    # Initialize the agent with tools
    llm = get_model()
    agent = create_react_agent(llm, tools, prompt=agent_prompt_template)
    
    # Prepare input for the agent
    agent_input = {
        "input": current_task,
        "chat_history": state["messages"][:1]  # Include only the original query
    }
    
    # Run the agent
    response = await agent.ainvoke(agent_input)
    
    # Return the agent's response
    return {
        "message": response["messages"][-1].content,
        "agent": agent_name,
        "task": current_task
    }

async def run_researcher(state: ParallelAgentState) -> Dict[str, Any]:
    """Run the researcher agent."""
    return await run_agent(state, "researcher", RESEARCHER_PROMPT)

async def run_writer(state: ParallelAgentState) -> Dict[str, Any]:
    """Run the writer agent."""
    return await run_agent(state, "writer", WRITER_PROMPT)

async def run_critic(state: ParallelAgentState) -> Dict[str, Any]:
    """Run the critic agent."""
    return await run_agent(state, "critic", CRITIC_PROMPT)

async def parallel_execution(state: ParallelAgentState) -> Dict[str, Dict[str, Any]]:
    """Execute multiple agents in parallel for each sub-task."""
    # Track all tasks and assign them to agents
    all_results = {}
    tasks = []
    
    for i, sub_task in enumerate(state["sub_tasks"]):
        # Assign each sub-task to one of our agents based on index
        if i % 3 == 0:
            agent_func = run_researcher
        elif i % 3 == 1:
            agent_func = run_writer
        else:
            agent_func = run_critic
            
        # Create a new state with just this sub-task
        task_state = {**state, "current_task": sub_task}
        
        # Create an async task for each agent
        task = asyncio.create_task(agent_func(task_state))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Organize results by task
    for result in results:
        all_results[result["task"]] = {
            "agent": result["agent"],
            "response": result["message"]
        }
    
    return {"results": all_results}

async def combine_results(state: ParallelAgentState) -> Dict[str, List[BaseMessage]]:
    """Combine results from all agents into a final response."""
    llm = get_model()
    
    # Prepare a summary of all agent results
    results_summary = ""
    for task, result in state["results"].items():
        results_summary += f"\nTask: {task}\nAgent: {result['agent']}\nResponse: {result['response']}\n"
    
    # Template for combining results
    prompt = PromptTemplate(
        template="""You are a synthesis agent that combines insights from multiple sources.
        
        Original query: {original_query}
        
        Results from parallel agents:
        {results_summary}
        
        Provide a comprehensive answer that integrates all relevant information from the different agents.
        The response should be well-structured and directly address the original query.
        """,
        input_variables=["original_query", "results_summary"]
    )
    
    # Process the results to generate a combined response
    chain = prompt | llm
    original_query = state["messages"][0].content
    response = await chain.ainvoke({
        "original_query": original_query, 
        "results_summary": results_summary
    })
    
    final_message = AIMessage(content=response.content)
    
    return {"messages": [state["messages"][0], final_message]}

def get_current_task(state: ParallelAgentState) -> Dict[str, Dict[str, str]]:
    """Get the next task from the sub_tasks list."""
    if not state["sub_tasks"]:
        return {"current_task": {"task": "No tasks available"}}
    
    # Get the first task (we'll process them one by one)
    current_task = state["sub_tasks"][0]
    remaining_tasks = state["sub_tasks"][1:]
    
    return {
        "current_task": current_task,
        "sub_tasks": remaining_tasks
    }

def create_parallel_agent_workflow():
    """Create and return the parallel agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(ParallelAgentState)
    
    # Add nodes
    workflow.add_node("split_task", split_task)
    workflow.add_node("parallel_execution", parallel_execution)
    workflow.add_node("combine_results", combine_results)
    
    # Add edges
    workflow.add_edge(START, "split_task")
    workflow.add_edge("split_task", "parallel_execution")
    workflow.add_edge("parallel_execution", "combine_results")
    workflow.add_edge("combine_results", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# Example usage
async def run_parallel_example(query: str):
    """Run an example query through the parallel agent workflow."""
    workflow = create_parallel_agent_workflow()
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "sub_tasks": [],  # Initialize empty lists and dictionaries
        "current_task": {},
        "results": {}
    }
    config = {"configurable": {"thread_id": "1"}} 
    
    # Run the workflow
    result = await workflow.ainvoke(initial_state, config)
    
    return result["messages"][-1].content

if __name__ == "__main__":
    import asyncio
    
    # Example query
    query = "Analyze the potential impact of quantum computing on cybersecurity over the next decade."
    
    # Run the example
    response = asyncio.run(run_parallel_example(query))
    print("\nFinal Response:")
    print(response)