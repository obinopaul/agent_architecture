import os
from typing import Dict, List, Any, Optional, Sequence, TypedDict, Annotated
import operator
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool, DuckDuckGoSearchResults
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from .env file

# Set up LLM
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

def get_model(model_name=DEFAULT_MODEL):
    """Get a chat model instance with the specified model name."""
    return ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)

# Define common tools
def get_common_tools():
    """Get a list of common tools for agents."""
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    youtube_tool = YouTubeSearchTool()
    # web_search_tool = DuckDuckGoSearchResults()
    web_search_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    
    return [wikipedia_tool, youtube_tool, web_search_tool]

# Base agent state (used in most architectures)
class AgentState(TypedDict):
    """Base state that includes conversation messages."""
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Extended agent state for more complex architectures
class ExtendedAgentState(AgentState):
    """Extended state with additional fields for complex architectures."""
    intermediate_steps: List[Dict[str, Any]]
    selected_agents: List[str]
    current_agent_idx: int
    final_response: Optional[str]

# Common system prompts
RESEARCHER_PROMPT = """You are a skilled research agent that specializes in finding information.
When given a question, break it down and search for relevant facts and data.
Provide detailed, factual responses and cite your sources when possible."""

WRITER_PROMPT = """You are a skilled writing agent that excels at clear communication.
Your goal is to take information and transform it into well-structured, coherent text.
Focus on clarity, organization, and making complex information accessible."""

CRITIC_PROMPT = """You are a critical thinking agent that evaluates information.
Your job is to review content, identify potential issues, logical fallacies, or missing information.
Provide constructive feedback aimed at improving accuracy and quality."""

PLANNER_PROMPT = """You are a strategic planning agent that helps organize complex tasks.
Given a problem, break it down into clear, actionable steps.
Consider dependencies between tasks and create an efficient plan of action."""

INTEGRATION_PROMPT = """You are an integration agent that synthesizes information from multiple sources.
Combine diverse inputs into a coherent whole, identifying connections and resolving contradictions.
Present a unified perspective that represents the combined knowledge."""

# Helper function to create a basic agent prompt template
def create_agent_prompt(system_message: str) -> ChatPromptTemplate:
    """Create a standard agent prompt template with the given system message."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(
            content=system_message,
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])