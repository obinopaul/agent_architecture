"""
Aggregator Multi-Agent Architecture

In this architecture, multiple input streams are combined by an agent into a single output.
Useful for gathering diverse information from different sources and synthesizing it.
"""

import asyncio
from typing import Dict, List, Any, Annotated, Sequence, TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from utils import get_model, get_common_tools, create_agent_prompt
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT, INTEGRATION_PROMPT

# Define the state for the aggregator agents workflow
class AggregatorAgentState(TypedDict):
    """State for aggregator agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    agent_responses: Dict[str, str]
    perspectives: List[Dict[str, Any]]
    final_response: str

# Node functions
def parse_query(state: AggregatorAgentState) -> Dict:
    """Parse the query and prepare it for distribution to multiple agents."""
    # Get the query
    query = state["messages"][-1].content
    
    return {
        "query": query,
        "agent_responses": {},
        "perspectives": []
    }

async def fact_finding_agent(state: AggregatorAgentState) -> Dict[str, str]:
    """Agent focused on gathering factual information."""
    query = state["query"]
    
    # Create the agent with specialized prompt
    llm = get_model()
    tools = get_common_tools()
    
    fact_prompt = ChatPromptTemplate.from_messages([
         SystemMessage(content="""You are a specialized fact-finding agent. Your role is to gather concrete, 
        verifiable information related to the query. Focus on statistics, dates, definitions, 
        and established facts from reliable sources. Avoid opinions and interpretations.
        
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
    FACT_PROMPT_MAIN = fact_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )

    agent = create_react_agent(model = llm, tools = tools, prompt = FACT_PROMPT_MAIN)
    
    # Execute the agent with the properly structured input
    response = await agent.ainvoke({
        "messages": [HumanMessage(content=query)],  # This is the key change
        "is_last_step": False,
        "remaining_steps": 5  # Arbitrary value, adjust as needed
    })
    
    return {
        "agent": "fact_finder",
        "response": response["messages"][-1].content
    }

async def perspective_agent(state: AggregatorAgentState) -> Dict[str, str]:
    """Agent focused on gathering diverse perspectives and interpretations."""
    query = state["query"]
    
    # Create the agent with specialized prompt
    llm = get_model()
    tools = get_common_tools()
    
    perspective_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a perspective-gathering agent. Your role is to identify and present
        different viewpoints, opinions, and interpretations related to the query. Focus on 
        presenting multiple sides of issues, different schools of thought, and competing theories.
        Ensure balance and fair representation of diverse perspectives.

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
    perspective_prompt = perspective_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(model = llm, tools = tools, prompt = perspective_prompt)
    
    # Execute the agent with the properly structured input
    response = await agent.ainvoke({
        "messages": [HumanMessage(content=query)],  # This is the key change
        # "is_last_step": False,
        # "remaining_steps": 5  # Arbitrary value, adjust as needed
    })
    
    return {
        "agent": "perspective_gatherer",
        "response": response["messages"][-1].content
    }

async def trend_analysis_agent(state: AggregatorAgentState) -> Dict[str, str]:
    """Agent focused on identifying trends and patterns."""
    query = state["query"]
    
    # Create the agent with specialized prompt
    llm = get_model()
    tools = get_common_tools()
    
    trend_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a trend analysis agent. Your role is to identify patterns, trends,
        and developments related to the query. Focus on how things have changed over time,
        emerging developments, and potential future directions. Look for connections and
        correlations between different factors.

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
    trend_prompt = trend_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(model = llm, tools = tools, prompt = trend_prompt)
    
    # Execute the agent with the properly structured input
    response = await agent.ainvoke({
        "messages": [HumanMessage(content=query)],  # This is the key change
        # "is_last_step": False,
        # "remaining_steps": 5  # Arbitrary value, adjust as needed
    })
    
    return {
        "agent": "trend_analyzer",
        "response": response["messages"][-1].content
    }

async def collect_responses(state: AggregatorAgentState) -> Dict:
    """Collect responses from multiple agents in parallel."""
    # Create tasks for each agent
    tasks = [
        asyncio.create_task(fact_finding_agent(state)),
        asyncio.create_task(perspective_agent(state)),
        asyncio.create_task(trend_analysis_agent(state))
    ]
    
    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    
    # Organize responses by agent
    agent_responses = {}
    for response in responses:
        agent_responses[response["agent"]] = response["response"]
    
    return {
        "agent_responses": agent_responses
    }

def extract_perspectives(state: AggregatorAgentState) -> Dict:
    """Extract and organize key perspectives from agent responses."""
    # Get the agent responses
    agent_responses = state["agent_responses"]
    
    # Use an LLM to extract key perspectives
    llm = get_model()
    
    extraction_prompt = PromptTemplate(
        template="""You are tasked with extracting key insights from multiple agent responses.
        
        For each agent response below, identify 3-5 key insights, facts, or perspectives that should
        be included in a comprehensive final response.
        
        Fact Finder's Response:
        {fact_finder_response}
        
        Perspective Gatherer's Response:
        {perspective_gatherer_response}
        
        Trend Analyzer's Response:
        {trend_analyzer_response}
        
        For each insight, include:
        1. The source agent
        2. A concise summary of the insight
        3. Relevance to the original query: {query}
        
        Format your response as a structured list of insights.
        """,
        input_variables=["fact_finder_response", "perspective_gatherer_response", 
                        "trend_analyzer_response", "query"]
    )
    
    # Execute the extraction
    response = llm.invoke(
        extraction_prompt.format(
            fact_finder_response=agent_responses.get("fact_finder", "No response"),
            perspective_gatherer_response=agent_responses.get("perspective_gatherer", "No response"),
            trend_analyzer_response=agent_responses.get("trend_analyzer", "No response"),
            query=state["query"]
        )
    )
    
    # Parse the response to get perspectives
    # In a real implementation, you might want to use a more structured approach
    perspectives = [
        {"content": line.strip()} 
        for line in response.content.split("\n") 
        if line.strip() and not line.strip().isdigit()
    ]
    
    return {
        "perspectives": perspectives
    }

def aggregate_node(state: AggregatorAgentState) -> Dict:
    """Aggregator agent that combines multiple inputs into a cohesive output."""
    # Get the agent responses and extracted perspectives
    agent_responses = state["agent_responses"]
    perspectives = state["perspectives"]
    query = state["query"]
    
    # Create the aggregator agent
    llm = get_model()
    
    aggregation_prompt = PromptTemplate(
        template="""You are an integration specialist tasked with synthesizing information from multiple sources
        into a comprehensive, cohesive response.
        
        Original Query: {query}
        
        You have received input from three specialized agents:
        
        1. Fact Finder
        {fact_finder_response}
        
        2. Perspective Gatherer
        {perspective_gatherer_response}
        
        3. Trend Analyzer
        {trend_analyzer_response}
        
        Key perspectives extracted:
        {perspectives}
        
        Create a comprehensive response that:
        1. Integrates information from all sources
        2. Resolves any contradictions or inconsistencies
        3. Provides a balanced view that includes facts, diverse perspectives, and trends
        4. Directly addresses the original query
        5. Is well-structured and flows logically
        
        Your response should synthesize, not just concatenate, the information from different sources.
        """,
        input_variables=["query", "fact_finder_response", "perspective_gatherer_response", 
                        "trend_analyzer_response", "perspectives"]
    )
    
    # Format perspectives for the prompt
    formatted_perspectives = "\n".join([f"- {p['content']}" for p in perspectives])
    
    # Execute the aggregation
    response = llm.invoke(
        aggregation_prompt.format(
            query=query,
            fact_finder_response=agent_responses.get("fact_finder", "No response"),
            perspective_gatherer_response=agent_responses.get("perspective_gatherer", "No response"),
            trend_analyzer_response=agent_responses.get("trend_analyzer", "No response"),
            perspectives=formatted_perspectives
        )
    )
    
    final_response = response.content
    
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)]
    }

def create_aggregator_agent_workflow():
    """Create and return the aggregator agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(AggregatorAgentState)
    
    # Add nodes
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("collect_responses", collect_responses)
    workflow.add_node("extract_perspectives", extract_perspectives)
    workflow.add_node("aggregate", aggregate_node)
    
    # Add edges
    workflow.add_edge(START, "parse_query")
    workflow.add_edge("parse_query", "collect_responses")
    workflow.add_edge("collect_responses", "extract_perspectives")
    workflow.add_edge("extract_perspectives", "aggregate")
    workflow.add_edge("aggregate", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())

# Example usage
async def run_aggregator_example(query: str):
    """Run an example query through the aggregator agent workflow."""
    workflow = create_aggregator_agent_workflow()
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    config = {"configurable": {"thread_id": "1"}} 
    
    # Run the workflow
    result = await workflow.ainvoke(initial_state, config)
    
    return {
        "agent_responses": result["agent_responses"],
        "perspectives": result["perspectives"],
        "final_response": result["final_response"]
    }

if __name__ == "__main__":
    import asyncio
    
    # Example query
    query = "What are the implications of artificial intelligence for the future of work?"
    
    # Run the example
    result = asyncio.run(run_aggregator_example(query))
    
    print("\n--- Agent Responses ---")
    for agent, response in result["agent_responses"].items():
        print(f"\n{agent.upper()}:")
        print(response[:300] + "..." if len(response) > 300 else response)
    
    print("\n--- Extracted Perspectives ---")
    for perspective in result["perspectives"][:5]:  # Show first 5 perspectives
        print(f"- {perspective['content']}")
    
    print("\n--- Final Aggregated Response ---")
    print(result["final_response"])