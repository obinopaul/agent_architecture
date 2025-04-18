Agentic Aechitecctures:
Single vs Multi-Agent Systems
Single-Agent Architecture
A user connects directly to one AI agent, which interfaces with memory and tools to perform tasks.
Multi-Agent Architecture
A user connects to a primary AI agent that coordinates with multiple other AI agents. Each agent can access specialized memory and tools. This creates a more complex but potentially more capable system.
Patterns in Multi-Agent Systems
Parallel
Multiple agents work simultaneously on tasks, with designated input and output points.
Sequential
Agents work in order, passing results from one to the next, with clear input and output points.
Loop
Similar to sequential but includes a feedback mechanism where output can be cycled back as input.
Router
One agent directs input to different output paths based on task requirements.
Aggregator
Multiple input streams are combined by an agent into a single output.
Network
A mesh-like configuration where multiple interconnected agents collaborate, with designated input and output points.
Hierarchical
A structured tree-like organization where a main agent delegates to subordinate agents.
Examples of Multi-Agent Architectures
Example 1: Hierarchical
A main agent receives user input and delegates to three specialized agents:

One handles vector search and vector database interactions
One manages web search
One manages tools and communication (Slack, Gmail)

Example 2: Human-in-the-Loop
The main agent coordinates between:

An agent that handles web search and vector search/database tasks
An agent that interacts with the user and manages tools (Slack, Gmail)

Example 3: Shared Tools
Multiple specialized agents share access to common tools:

Two agents handle vector search and database operations
One agent manages web search capabilities

Example 4: Sequential
A linear chain of three AI agents working in sequence, with the first agent also connecting to vector search and database, and the second agent connecting to web search.
Example 5: Shared Database with Different Tools
Agents share access to a central data transformation resource:

One agent handles web search
One agent manages vector search and database operations
Both coordinate through the shared data transformation component

Example 6: Memory-Transformation Through Tool Use
A system where tools transform how information is stored and retrieved:

Main agent connects to web search, vector search, and memory
Secondary agent manages data transformation processes
Both share access to the transformed data





Example:
"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
import os 
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END


from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from datetime import date
import logging

# Suppress debug messages from ipywidgets
logging.getLogger('ipywidgets').setLevel(logging.WARNING)
logging.getLogger('comm').setLevel(logging.WARNING)
logging.getLogger('tornado').setLevel(logging.WARNING)
logging.getLogger('traitlets').setLevel(logging.WARNING)

#------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Disable all logging globally
logging.disable(logging.CRITICAL)  # Disable all logging below CRITICAL level

# Redirect all logging output to os.devnull
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])

# Suppress warnings as well (optional)
import warnings
warnings.filterwarnings("ignore")

from agents.nba.agents import game_supervisor, player_supervisor, teams_supervisor

from typing import List, Dict, Any, Optional, Sequence, TypedDict, Annotated
import pandas as pd
from datetime import datetime, timedelta
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
# from langgraph.prebuilt import create_function_calling_executor  # Correct import
from langgraph.graph import END, StateGraph
import operator
import asyncio
from core import get_model, settings 


# --- AgentState Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sub_queries: Optional[List[Dict[str, str]]] = None  # List of dicts: {query, supervisor}
    current_query: Optional[str] = None
    # agent_index: Not needed - we're using supervisor names directly


# --- Supervisor Prompts (from previous response - included for completeness) ---
# (Include the GAME_SUPERVISOR_PROMPT, PLAYER_SUPERVISOR_PROMPT,
#  TEAM_SUPERVISOR_PROMPT, and NBA_SUPERVISOR_PROMPT from a previous response here.)
#  For brevity, I am not putting it again.

# --- Helper Functions ---

async def split_node(state: AgentState) -> Dict[str, List[Dict[str, str]]]:
    """Splits the user's query into sub-queries and assigns supervisors."""

    class SubQuery(BaseModel):
        query: str = Field(..., description="The sub-query text.")
        supervisor: str = Field(..., description="The name of the assigned supervisor agent ('game_supervisor', 'player_supervisor', or 'teams_supervisor').")

    class ParsedOutput(BaseModel):
        sub_queries: List[SubQuery] = Field(..., description="A list of sub-query dictionaries.  Each dictionary MUST contain a 'query' key (the sub-query) and a 'supervisor' key (the name of the assigned supervisor). The number of sub-queries should be between 1 and 7, inclusive, based on the complexity of the original query.  Simple queries should have fewer sub-queries.")


    llm = get_model(settings.DEFAULT_MODEL)
        
    llm_with_structure = llm.with_structured_output(ParsedOutput)

    prompt = PromptTemplate(
        template="""You are an advanced query decomposition and routing assistant for an NBA information system. Your task is to analyze a user's query and break it down into a set of smaller, focused sub-queries, assigning each to the most appropriate supervisor agent.

        **Analyze the user's query:** `{query}`.

        **Determine the optimal number of sub-queries:** Break down the query into *1 to 7* sub-queries, depending on the complexity and scope of the original question.
            *   A simple question requiring information from only one area might only need *one* sub-query.
            *   A complex question requiring information from multiple areas, or requiring in-depth analysis, might need *several* sub-queries, *but no more than 7*.  Use your judgment and **prioritize simplicity and directness**.
            *   **Do not generate unnecessary sub-queries.** If the original query can be answered directly by one supervisor, do *not* create additional sub-queries.

        **Supervisor Agents and Their Expertise:**

        1.  **game_supervisor**: This supervisor is the expert on all things related to NBA *games*.  It can handle queries about:
            *   **Live Games:**  Current scores, real-time statistics, play-by-play information.
            *   **Game Schedules:**  Past, present, and future game schedules, including dates, times, and opponents.
            *   **Team Game Logs:**  Detailed historical records of a specific team's games for a given season, including results (wins/losses), opponents, and dates. It does *not* handle individual player stats within those games (that's for the player_supervisor).
            *   **General Game Information:** Overall league standings, playoff brackets.  It is *not* the best choice for in-depth analysis of team *strategy* or *ownership* (that would be teams_supervisor).

        2.  **player_supervisor**: This supervisor is the expert on all things related to individual NBA *players*. It can handle queries about:
            *   **Player Biographical Information:** Height, weight, date of birth, current team, position, jersey number, college, etc.
            *   **Player Career Statistics:**  Points, rebounds, assists, steals, blocks, field goal percentage, three-point percentage, free throw percentage, etc. (per game, totals, career averages).  This includes regular season, playoffs, and All-Star games.
            *   **General Player Information**: This can include using web search to retrieve player news.

        3.  **teams_supervisor**: This supervisor is the expert on all things related to NBA *teams* (but *not* individual games, which are handled by the game_supervisor). It can handle queries about:
            *   **Team Game Logs:** A specific team's game history for a given season.  This includes dates, opponents, and results (wins/losses).
            * **General Team Information (via web search):**  Team news, ownership, coaching staff, arena information, and other details not directly related to game logs or individual player stats.
            *   **Team Statistics**: Overall league standings, specific team stats for a season (or multiple seasons), comparisons between teams, team records, and conference/division standings.


        **Output Format:**

        Return a JSON object with a single key, 'sub_queries'. The value is a list of dictionaries. Each dictionary MUST have the following keys:

        *   `query`: The sub-query text (string).
        *   `supervisor`: The name of the supervisor agent that should handle this sub-query (string). Must be one of: "game_supervisor", "player_supervisor", or "teams_supervisor".

        **Key Principles:**

        *   **Directness:**  Sub-queries should be as direct and to-the-point as possible.
        *   **Specificity:** Each sub-query must be clearly answerable by *one* of the supervisors.
        *   **Independence:** Sub-queries should be as independent of each other as possible.
        *   **Simplicity:**  Favor fewer sub-queries when possible.  Avoid unnecessary decomposition.

        **EXAMPLES:**

        **Example 1 (Simple - Single Supervisor):**
        User Query: "What's the score of the Lakers game?"
        Output: `{{"sub_queries": [{{"query": "What's the score of the Lakers game right now?", "supervisor": "game_supervisor"}}]}}`

        **Example 2 (Intermediate - Two Supervisors):**
        User Query: "What's LeBron James' height and current team?"
        Output: `{{"sub_queries": [{{"query": "What is LeBron James' height?", "supervisor": "player_supervisor"}}, {{"query": "What is LeBron James' current team?", "supervisor": "player_supervisor"}}]}}`

        **Example 3 (Advanced - Mixed Supervisors - Live and Historical Data):**
        User Query: "Is LeBron James playing tonight? If so, what are his career playoff averages?"
        Output: `{{"sub_queries": [{{"query": "Is LeBron James playing in tonight's game?", "supervisor": "game_supervisor"}}, {{"query": "What are LeBron James' career playoff averages?", "supervisor": "player_supervisor"}}]}}`
        
        **Example 4 (Advanced - Multiple Supervisors, but still concise):**
        User Query: "Compare the average points per game for LeBron James and Stephen Curry over the last three seasons, and tell me which team had the best record in their conference last season."
        Output: `{{"sub_queries": [{{"query": "What were LeBron James' average points per game for the last three seasons?", "supervisor": "player_supervisor"}}, {{"query": "What were Stephen Curry's average points per game for the last three seasons?", "supervisor": "player_supervisor"}}, {{"query": "Which NBA team had the best record in their conference last season?", "supervisor": "teams_supervisor"}}]}}`

        **Example 5 (Advanced - Mixed Supervisors with Analysis):**
        User Query: "Analyze the impact of Draymond Green's defensive presence on the Golden State Warriors' win percentage over the past three seasons.  Also, what is his average blocks per game?"
        Output:  `{{"sub_queries": [{{"query": "What was the Golden State Warriors' win percentage over the past three seasons?", "supervisor": "teams_supervisor"}}, {{"query": "What is Draymond Green's average blocks per game over the past three seasons?", "supervisor": "player_supervisor"}}, {{"query": "How does Draymond Green's presence/absence correlate with the Warriors' win percentage over the past three seasons?", "supervisor": "teams_supervisor"}}]}}`

        **Example 6 (Advanced - Statistical Trends and Comparisons):**
        User Query: "Compare the assist-to-turnover ratio of Chris Paul, Rajon Rondo, and Russell Westbrook over their entire careers, and analyze how their passing styles have evolved."
        Output: `{{"sub_queries": [{{"query": "What is Chris Paul's career assist-to-turnover ratio?", "supervisor": "player_supervisor"}}, {{"query": "What is Rajon Rondo's career assist-to-turnover ratio?", "supervisor": "player_supervisor"}}, {{"query": "What is Russell Westbrook's career assist-to-turnover ratio?", "supervisor": "player_supervisor"}}, {{"query": "How has Chris Paul's passing style evolved over his career?", "supervisor": "player_supervisor"}}, {{"query": "How has Rajon Rondo's passing style evolved over his career?", "supervisor": "player_supervisor"}}, {{"query": "How has Russell Westbrook's passing style evolved over his career?", "supervisor": "player_supervisor"}}]}}`

        **Example 7 (Advanced - Game Strategy and Outcomes):**
        User Query: "Analyze the effectiveness of different defensive schemes (e.g., switching, hedging, dropping) against pick-and-roll plays involving Stephen Curry and Draymond Green.  Which scheme leads to the lowest points per possession for the opposing team, and how does this vary based on the personnel on the court?"
        Output: `{{"sub_queries": [{{"query": "What are the different defensive schemes used against pick-and-roll plays involving Stephen Curry and Draymond Green?", "supervisor": "game_supervisor"}}, {{"query": "What is the effectiveness (points per possession allowed) of switching defenses against Curry/Green pick-and-rolls?", "supervisor": "game_supervisor"}}, {{"query": "What is the effectiveness of hedging defenses against Curry/Green pick-and-rolls?", "supervisor": "game_supervisor"}}, {{"query": "What is the effectiveness of dropping defenses against Curry/Green pick-and-rolls?", "supervisor": "game_supervisor"}}, {{"query": "How does the effectiveness of different defensive schemes against Curry/Green pick-and-rolls vary based on opposing personnel?", "supervisor": "game_supervisor"}}]}}`
        
        """,
        input_variables=["query"]
    )

    chain = prompt | llm_with_structure
    query = state["messages"][-1].content
    structured_output = await chain.ainvoke({"query": query})
    return {"sub_queries": [sub_query.dict() for sub_query in structured_output.sub_queries]}  # Convert to dict



async def run_supervisor(state: AgentState, supervisor_dict: Dict[str, Any]) -> AgentState:
    """Runs the appropriate supervisor based on the assigned supervisor name."""
    sub_query_info = state['current_query']
    sub_query = sub_query_info['query']
    supervisor_name = sub_query_info['supervisor']
    current_date = datetime.now().isoformat()

    # --- CRITICAL CHANGE:  Look up the supervisor by NAME ---
    supervisor = supervisor_dict[supervisor_name]

    supervisor_input = {
        "messages": state["messages"][:1] + [HumanMessage(content=f"{sub_query} Today is: {current_date}")],
    }
    response = await supervisor.ainvoke(supervisor_input)
    return {"messages": [response['messages'][-1]]}


async def parallel_runner(state: AgentState, supervisor_dict: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
    """Runs the appropriate supervisors in parallel for each sub-query."""
    tasks = []
    for sub_query_info in state["sub_queries"]:
        # Pass the entire dictionary containing query AND supervisor
        updated_state = {**state, "current_query": sub_query_info}
        task = asyncio.create_task(run_supervisor(updated_state, supervisor_dict))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    all_messages = []
    for result in results:
        all_messages.extend(result["messages"])
    return {"messages": all_messages}



async def combine_results(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Combines results and presents to LLM for final answer."""
    final_results = [msg.content for msg in state["messages"][1:]]
    combined_results_str = "\n\n".join(final_results)

    final_llm = get_model(settings.DEFAULT_MODEL)
        
    final_prompt = PromptTemplate(
        template="""You are an expert NBA assistant.

        Original query: {original_query}

        Sub-query results: {combined_results}

        Provide a comprehensive answer.
        """,
        input_variables=["original_query", "combined_results"]
    )
    final_chain = final_prompt | final_llm
    final_answer = await final_chain.ainvoke({"original_query": state["messages"][0].content, "combined_results": combined_results_str})

    new_messages = [state["messages"][0], HumanMessage(content=final_answer.content)]
    return {"messages": new_messages}



supervisor_dict = {
    "game_supervisor": game_supervisor,
    "player_supervisor": player_supervisor,
    "teams_supervisor": teams_supervisor,
}
workflow = StateGraph(AgentState)
workflow.add_node("split_query", split_node)
workflow.add_node("parallel_supervisors", lambda state, supervisors=supervisor_dict: asyncio.run(parallel_runner(state, supervisors))) # Changed to dict
workflow.add_node("combine_results", combine_results)

workflow.add_edge(START, "split_query")
workflow.add_edge("split_query", "parallel_supervisors")
workflow.add_edge("parallel_supervisors", "combine_results")
workflow.add_edge("combine_results", END)

nba_graph = workflow.compile(name="NBA_Workflow",
                           checkpointer=MemorySaver())



# initial_state = {
#     "messages": [HumanMessage(content="Tell me about the Rockets vs Pacers game happening today. What are the key stats to look out for?")]
# }
# final_state = app_nba.ainvoke(initial_state)






example:
from datetime import datetime
from typing import cast

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel, system_prompt: SystemMessage
) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one paragraph summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is to demonstrate doing work before the interrupt"""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    birthdate: datetime | None = Field(
        description="The extracted birthdate. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


async def determine_birthdate(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node examines the conversation history to determine user's birthdate.  If no birthdate is found, it will perform an interrupt before proceeding."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    )
    response = await model_runnable.ainvoke(state, config)
    response = cast(BirthdateExtraction, response)

    # If no birthdate found, interrupt
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\n" "Please tell me your birthdate?")
        # Re-run extraction with the new input
        state["messages"].append(HumanMessage(birthdate_input))
        return await determine_birthdate(state, config)

    # Birthdate found
    return {
        "birthdate": response.birthdate,
    }


sign_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
What is the sign of somebody born on {birthdate}?
""")


async def determine_sign(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node determines the zodiac sign of the user based on their birthdate."""
    if not state.get("birthdate"):
        raise ValueError("No birthdate found in state")

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, sign_prompt.format(birthdate=state["birthdate"].strftime("%Y-%m-%d"))
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("determine_sign", determine_sign)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "determine_sign")
agent.add_edge("determine_sign", END)

interrupt_agent = agent.compile(
    checkpointer=MemorySaver(),
)
interrupt_agent.name = "interrupt-agent"





example:
from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

research_assistant = agent.compile(checkpointer=MemorySaver())





> AI Agents


'''

AI Agent Architectures
├── Profiling Module or Perception Module (The Eyes and Ears of the Agent)
│   ├── Sensory expertise
│   ├── Perceives and interprets the environment and communicated with other agents.
│   ├──  --> the agent may collect and analyze information from its environment like how human senses work.
│   ├──  --> helps it comprehend visual signals, recognize speech patterns, and sense tactile inputs..
│   └── Example: Recognizing objects via sensors in self-driving cars
│
├── Memory Module
│   ├── Stores data, rules, and patterns
│   ├── Enables knowledge recall and decision-making
│   └── Example: Chatbots recalling customer preferences
│   
├── Planning Module
│   ├── Analyzes current situations
│   ├── Strategizes actions to meet goals
│   └── Example: Optimizing delivery routes
│
├── Action Module
│   ├── Executes planned actions
│   ├── Interfaces with external systems
│   └── Example: Robotic arms assembling parts
│
├── Learning Module
│   ├── Adapts and improves performance
│   ├── Methods include:
│   │   ├── Supervised Learning
│   │   ├── Unsupervised Learning
│   │   └── Reinforcement Learning
│   └── Example: Agents learning optimal decisions from feedback
│
├── Data Structuring and Transformation Module
│   ├── Organizes and preprocesses data both from the environment as well as from the memory module
│   ├── Converts data into trainable formats
│   └── Example: Formatting images for neural network training
│
├── Training Module
│   ├── Performs training operations and updates
│   ├── Use methods like:
│   │   ├── Supervised, Unsupervised, and Reinforcement Learning
│   │   ├── Computer Vision, LLM, Time series Learning
│   │   ├── ANN, CNN, Transformers, GANs, GNNs, 
│   │   └── Tools like TensorFlow, PyTorch, Keras, Scikit-learn
│   └── Example: AI training itself in virtual environments
│
└── Other Modules

 
'''


> requirements.txt
'''
langchain_community
tiktoken
langchainhub
langchain
chromadb
langgraph
tavily-python
python-dotenv
google-generativeai
langchain_google_genai
langchain-nomic
langchain-text-splitters
langchain_mistralai
wikipedia
langchain_huggingface
google-search-results
faiss-cpu
sentence-transformers
youtube-search


'''
> AI Agents Tips
# RAG (Retrieval Augmented Generation) 
    # LangChain: 
    # RAGFlow by infiniflow: https://github.com/infiniflow/ragflow
    # Haystack by deepset-ai: https://github.com/deepset-ai/haystack
    # txtai by neuml: https://github.com/neuml/txtai
    # LLM-App by pathwaycom: https://github.com/pathwaycom/llm-app 

# Intelligent Document Processing
    # LangChain
    # docling: https://github.com/docling-project/docling
    # marker: https://github.com/VikParuchuri/marker
    # unstructured: https://github.com/Unstructured-IO/unstructured
    # landingAI document extractor: https://github.com/landing-ai/agentic-doc
    # Haystack by deepset-ai: https://github.com/deepset-ai/haystack
    # Microsoft:
        # Markitdown: https://github.com/microsoft/markitdown
        # Markitdown mcp server: https://github.com/microsoft/markitdown/tree/main/packages/markitdown-mcp
        # Azure Document Intelligence: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/how-to-guides/create-document-intelligence-resource?view=doc-intel-4.0.0
        
# LLM Chatbots (Langchain and LangGraph with AI Agents)
   [Good]# onyx: https://github.com/onyx-dot-app/onyx
   # OpenWebUI:
   # Open MCP Client: https://github.com/CopilotKit/open-mcp-client
   # LangGraph Agent Chat: https://github.com/langchain-ai/agent-chat-ui
   # dify: https://github.com/langgenius/dify
   # anything-llm: https://github.com/Mintplex-Labs/anything-llm

# AI-Powered Knowledge Bases


# Customer Service Automation 


# AI Agents
> n8n
# n8n interface

# n8n is a workflow automation tool that enables you to connect your favorite apps, services, and devices.
# It allows you to automate workflows and integrate your apps, services, and devices with each other.

    # workflow: a sequence of connected steps that automate a process.
    # node: a single step in a workflow.
    # connection: a link between two nodes that passes data from one node to another.
    # execution: a single run of a workflow.

# Types of Nodes
    # Trigger Node: The starting point of a workflow. It initiates the execution of a workflow.
    # Regular Node: A node that performs a specific action or operation.
    # Parameter Node: A node that stores and provides data to other nodes in the workflow.
    # Sub-Workflow Node: A node that allows you to reuse a workflow within another workflow.
    # Webhook Node: A node that receives data from an external service or application.
    # Error Node: A node that handles errors that occur during the execution of a workflow.
    # No-Operation Node: A node that does nothing. It is used for debugging and testing purposes.
    
    # OR
    # Trigger Nodes: These nodes initiate the execution of a workflow. They are the starting points of a workflow.
    # Data Transformation Nodes: These nodes perform operations on data. They transform, filter, or manipulate data in some way.
    # Action Nodes: These nodes perform actions such as sending an email, making an API call, or updating a database.
    # Logic Nodes: These nodes control the flow of a workflow. They make decisions based on conditions and determine the path a workflow should take.




## Langchain Tools
### Custom Tools 
from langchain_community.tools import YouTubeSearchTool, WikipediaSummaryTool, CustomTool
from langchain_community.tools.tavily_search_tool import TavilySearchResults, TavilyAnswer
from langchain.agents import tool

tool_1 = YouTubeSearchTool()
tool_2 = WikipediaSummaryTool()
tool_3 = TavilySearchResults()

@tool
def get_word_length(word: str) -> int:
    """Return the length of a word."""
    return len(word)

print(f'Length of the word '{get_word_length.invoke("hello")})

print(get_word_length.name)
print(get_word_length.description)
print(get_word_length.args)
#----------------------------------------------------------------------------------------
### Custom Tools 
from langchain_community.tools import YouTubeSearchTool, WikipediaSummaryTool, CustomTool
from langchain_community.tools.tavily_search_tool import TavilySearchResults, TavilyAnswer

tool_1 = YouTubeSearchTool()
tool_2 = WikipediaSummaryTool()
tool_3 = TavilySearchResults()

tools = [tool_1, tool_2, tool_3]

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
from langchain.agents import tool

@tool
def get_word_length(text: str) -> int:
    """Return the length of a word."""
    return len(text)

print(get_word_length.invoke("hello"))

print(get_word_length.name)
print(get_word_length.description)
print(get_word_length.args)


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

google_search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Web Answer",
        func = google_search.run,
        description="Get an intermediate answer to a question.",
        verbose = True
    )
]


#-----------------------------------------Custom Tool from a LangChain Chain ----------------------------------------
#--------------------------------------------------------------------------------------------------------------------

from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

@tool("query_param_checker")
def query_param_checker(user_query: str, generated_query_params: str) -> str:
    """
    This tool checks if the query parameters generated by query_param_generator are valid.
    It uses an LLM to evaluate the parameters based on a simple prompt.
    """
    # Define a simple prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Check if the generated query parameters are valid for the user query."),
            ("human", "User Query: {user_query}\nGenerated Query Parameters: {generated_query_params}"),
        ]
    )

    # Create a chain with the prompt and LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm

    # Invoke the chain with the inputs
    result = chain.invoke({"user_query": user_query, "generated_query_params": generated_query_params})

    # Return the LLM's response
    return result.content


#-----------------------------Custom Tool from Class (RECOMMENDED 1) ------------------------------------------
#----------------------------------------------------------------------------------------------

# Define Input Schema
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

# Define the Tool
class TavilySearchTool:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a web search using the Tavily search engine.
        """
        try:
            # Initialize the Tavily search tool with the configured max_results
            search_tool = TavilySearchResults(max_results=self.max_results)

            # Perform the search
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

# Create the LangChain Tool
search_tool = Tool(
    name="Tavily Search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)


#-----------------------------Structured Tool from Class (BEST AND MOST RECOMMENDED) ------------------------------------------
#----------------------------------------------------------------------------------------------
from langchain.tools.base import StructuredTool

# Define Input Schema
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

# Define the Tool
class TavilySearchTool:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a web search using the Tavily search engine.
        """
        try:
            # Initialize the Tavily search tool with the configured max_results
            search_tool = TavilySearchResults(max_results=self.max_results)

            # Perform the search
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

# Create the LangChain Tool
search_tool = StructuredTool(
    name="Tavily Search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)


# ------------------------------------------------ Convert Tool to Structured Tool ------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
from langchain.tools.base import StructuredTool
from langchain.agents import Tool, load_tools
from langchain_core.tools import StructuredTool

def convert_to_structured_tool(tool):
    return StructuredTool.from_function(tool.func, name=tool.name, description=tool.description)

tools = load_tools(['serpapi'])
tools = [convert_to_structured_tool(tool) for tool in tools]


#---------------------------------- Custom Tool (RECOMMENDED 2) -------------------------------------------------
#------------------------------------------------------------------------------------------------
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Union
import requests
import os

# Define Input Schema: Use Pydantic to define input parameters and descriptions for your tool.
class MyToolInput(BaseModel):
    param1: str = Field(..., description="Description of param1.")
    param2: int = Field(default=10, description="Description of param2.")
    
# Create the Tool: Use the @tool decorator to define a custom tool.
@tool("my_tool_function", args_schema=MyToolInput, return_direct=True)
def my_tool_function(param1: str, param2: int = 10) -> Union[Dict, str]:
    """
    Description of what the tool does.
    """
    try:
        url = (
            f'https://api.financialdatasets.ai/insider-transactions'
            f'?ticker={param1}'
            f'&limit={param2}'
            )
        # Perform the task (e.g., call an API, process data, etc.)
        response = requests.get(url, headers={'X-API-Key': api_key})
        return response
    except Exception as e:
        return {"error": str(e)}

tools = [my_tool_function, annual_report_tool, get_word_length]


#-----------------------------Custom Tool from a Custom Chain----------------------------------
#----------------------------------------------------------------------------------------------

from langchain.chains.base import Chain
from typing import Dict, List

class AnnualReportChain(Chain):
    chain: Chain

    @property
    def input_keys(self) -> List[str]:
        return list(self.chain.input_keys)

    @property
    def output_keys(self) -> List[str]:
        return ['output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Queries the database to get the relevant documents for a given query
        query = inputs.get("input_documents", "")
        docs = vectorstore.similarity_search(query, include_metadata=True)
        output = chain.run(input_documents=docs, question=query)
        return {'output': output}
    
    

from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Initialize your custom Chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
chain = load_qa_chain(llm)
annual_report_chain = AnnualReportChain(chain=chain)

# Initialize your custom Tool
annual_report_tool = Tool(
    name="Annual Report",
    func=annual_report_chain.run,
    description="""
    useful for when you need to answer questions about a company's income statement,
    cash flow statement, or balance sheet. This tool can help you extract data points like
    net income, revenue, free cash flow, and total debt, among other financial line items.
    """
)



#---------------------------------- Creating a Node from Tools ----------------------------------
#------------------------------------------------------------------------------------------------
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START

builder = StateGraph(State)
tool_node = ToolNode(tools=tools)
builder.add_node("tools", tool_node)


# ----------------------------------------- Vision Agent Tool -------------------------------------
# -----------------------------------------------------------------------------------------------
from vision_agent.agent import VisionAgentCoderV2
from vision_agent.models import AgentMessage

agent = VisionAgentCoderV2(verbose=True)
code_context = agent.generate_code(
    [
        AgentMessage(
            role="user",
            content="Count the number of people in this image",
            media=["people.png"]
        )
    ]
)

with open("generated_code.py", "w") as f:
    f.write(code_context.code + "\n" + code_context.test)
    
# --------------Using the tools directly in the code---------------------
import vision_agent.tools as T
import matplotlib.pyplot as plt

image = T.load_image("people.png")
dets = T.countgd_object_detection("person", image)
# visualize the countgd bounding boxes on the image
viz = T.overlay_bounding_boxes(image, dets)

# save the visualization to a file
T.save_image(viz, "people_detected.png")

# display the visualization
plt.imshow(viz)
plt.show()


# ------------------------------Using the tools on video files---------------------
import vision_agent.tools as T

frames_and_ts = T.extract_frames_and_timestamps("people.mp4")
# extract the frames from the frames_and_ts list
frames = [f["frame"] for f in frames_and_ts]

# run the countgd tracking on the frames
tracks = T.countgd_sam2_video_tracking("person", frames)
# visualize the countgd tracking results on the frames and save the video
viz = T.overlay_segmentation_masks(frames, tracks)
T.save_video(viz, "people_detected.mp4")
> LangChain, LangGraph, and LangSmith
# LangChain 
    # Tools
    # Agents
    # Chains
    # Multi-Agent Systems
    # Plan and Execute
    # Reflection and Learning
    # Communication
    # Perception

# LangChain is a platform that enables developers to build, test, and deploy blockchain applications using multiple programming languages.
# It provides a set of tools and libraries that simplify the development process and make it easier to create blockchain applications.



# Types of LangChain Agents
    # LangChain offers several agentic patterns, each tailored to specific needs. These include:

    # Tool Calling Agents: Designed for straightforward tool usage.
    # React Agents: Use reasoning and action mechanisms to dynamically decide the best steps.
    # Structured Chat Agents: Parse inputs and outputs into structured formats like JSON.
    # Self-Ask with Search: Handle queries by splitting them into smaller, manageable steps.
## Langchain Agent
> Create Tool Calling Agent
from langchain.agents import create_tool_calling_agent
from langchain.agents.tool_calling_agent import base
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain_openai import ChatOpenAI as LangchainChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_community.tools import YouTubeSearchTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor
import os

# Load API key
api_key = os.getenv("DEEPSEEK_API_KEY")

# Prompt
prompt = hub.pull("hwchase17/openai-functions-agent")
# or
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # First put the history
        ("placeholder", "{chat_history}"),
        # Then the new input
        ("human", "{input}"),
        # Finally the scratchpad
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Tools
tool_1 = YouTubeSearchTool()
tool_2 = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tool_3 = TavilySearchResults(max_results=10)

tools = [tool_1, tool_2, tool_3]

# LLM
llm = LangchainChatDeepSeek(
            api_key=api_key,
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
        )

# Agent

# Create a tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)
# agent = base.create_tool_calling_agent()

# Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=False,  # Only final output. If True, returns all intermediate steps
    handle_parsing_errors=True,  # Graceful parsing errors
)
        

query = input("Enter your query: ")

response = agent_executor.invoke(
    {
        "input": [HumanMessage(content=query)]
    }
)
> ReAct Agent

# The ReActAgent employs the ReAct (Reason+Act) framework, enabling the agent to perform both reasoning and actions within a 
# single framework. It integrates chain-of-thought reasoning with action execution, allowing the agent to handle complex, 
# multi-step tasks effectively.

from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from typing import Annotated

template = '''Answer the following questions as best as you can. You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation sequence can be repeated N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here is an example of how to use the tools:
Question: Generate a chart of the Fibonacci sequence.
Thought: I need to write Python code to generate the Fibonacci sequence and plot it.
Action: python_repl_tool
Action Input: 
```python
import matplotlib.pyplot as plt

def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

n = 10  # Number of Fibonacci numbers to generate
fib_sequence = fibonacci(n)
plt.plot(fib_sequence)
plt.title("Fibonacci Sequence")
plt.show()
```
Observation: The chart was successfully generated.
Thought: I now know the final answer.
Final Answer: The chart of the Fibonacci sequence has been generated.

Begin!
Question: {input}
Thought: {agent_scratchpad}
'''



repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str
    
prompt = PromptTemplate.from_template(template)
search_agent = create_react_agent(llm, tools = [python_repl_tool], prompt=prompt)

agent_executor = AgentExecutor(agent=search_agent, tools=[python_repl_tool], verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)
agent_executor.invoke({"input": "create a visualization of some advanced dataset."})
# agent_executor.invoke({"input": [HumanMessage(content="What is the capital of France?")]})
> LangGraph ReAct Agent
""" 
LangGraph's prebuilt create_react_agent does not take a prompt template directly as a parameter, but instead takes a prompt parameter. 
This modifies the graph state before the llm is called, and can be one of four values:

    1. A SystemMessage, which is added to the beginning of the list of messages.
    2. A string, which is converted to a SystemMessage and added to the beginning of the list of messages.
    3. A Callable, which should take in full graph state. The output is then passed to the language model.
    4. Or a Runnable, which should take in full graph state. The output is then passed to the language model.

"""

from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage, AgentMessage, AgentMessageWithScratchpad

# ----------------------------------------System Message-------------------
system_message = "You are a helpful assistant. Respond only in Spanish."
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

langgraph_agent_executor = create_react_agent(model, tools, prompt=system_message)
# -------------------------------------------------------------------------------------------------


# ----------------------------------------ChatPromptTemplate-------------------
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         # First put the history
#         ("placeholder", "{chat_history}"),
#         # Then the new input
#         ("human", "{input}"),
#         # Finally the scratchpad
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

react_agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an advanced AI assistant designed to solve complex tasks using a systematic, step-by-step approach. 

CORE AGENT INSTRUCTIONS:
1. ALWAYS follow the React (Reasoning and Acting) paradigm
2. For EACH task, you must:
   a) REASON about the problem
   b) DETERMINE which TOOL to use
   c) Take ACTION using the selected tool
   d) OBSERVE the results
   e) REFLECT and decide next steps

AVAILABLE TOOLS:
{tools}

TOOL USAGE PROTOCOL:
- You have access to the following tools: {tool_names}
- BEFORE using any tool, EXPLICITLY state:
  1. WHY you are using this tool
  2. WHAT specific information you hope to retrieve
  3. HOW this information will help solve the task

TOOL INTERACTION FORMAT:
When using a tool, you MUST follow this strict format:
Thought: [Your reasoning for using the tool]
Action: [Exact tool name]
Action Input: [Precise input for the tool]

After receiving the observation, you will:
Observation: [Tool's response]
Reflection: [Analysis of the observation and next steps]

FINAL OUTPUT EXPECTATIONS:
- Provide a comprehensive, step-by-step solution
- Cite sources and tools used
- Explain your reasoning at each stage
- Offer clear conclusions or recommendations

Are you ready to solve the task systematically and intelligently?"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{messages}"),
])

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Partial the prompt with tools and tool names
prompt = react_agent_prompt.partial(
    tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools])
)

# Create the React agent
agent = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=prompt
)

query = "Calculate the total cost of 15 items priced at $24.50 each, including a 7% sales tax"

messages = agent.invoke({"messages": [("human", query)]})


# -------------------------------------- STREAM MODE --------------------------------------
# Create the React agent
langgraph_agent_executor = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=prompt
)


for step in langgraph_agent_executor.stream(
    {"messages": [("human", query)]}, stream_mode="updates"
):
    print(step)
    
    
# --------------------------- FOR mAX ITERATION, USE RECURSION LIMIT ---------------------------
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

RECURSION_LIMIT = 2 * 3 + 1

langgraph_agent_executor = create_react_agent(model, tools=tools)

try:
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
except GraphRecursionError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
    
    

# --------------------------- FOR early_stopping_method ---------------------------
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

RECURSION_LIMIT = 2 * 1 + 1

langgraph_agent_executor = create_react_agent(model, tools=tools)

try:
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
except GraphRecursionError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
> Self Ask with Search Agent
# This agent incorporates a self-asking mechanism combined with search capabilities. It can autonomously formulate internal queries 
# to gather additional information necessary to answer user questions comprehensively

from langchain.agents import create_self_ask_with_search_agent
from langchain import hub
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

google_search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Web Answer",
        func = google_search.run,
        description="Get an intermediate answer to a question.",
        verbose = True
    )
]

prompt = hub.pull("hwchase17/self-ask-with-search")
search_agent = create_self_ask_with_search_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=search_agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)
agent_executor.invoke({"input": "What is the capital of France?"})
# agent_executor.invoke({"input": [HumanMessage(content="What is the capital of France?")]})
> StructuredChatAgent (can use multiple inputs)
# param format_instructions: str = 'Use a json blob to specify a tool by providing an action key (tool name) and an action_input 
# key (tool input).\n\nValid "action" values: "Final Answer" or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, 
# as shown:
# \n
# \n```
# \n{{{{
    # \n  "action": $TOOL_NAME,
    # \n  "action_input": $INPUT
    # \n}}}}
    # \n```
# 
# \n\nFollow this format:
# \n\nQuestion: input question to answer
# \nThought: consider previous and subsequent steps
# \nAction:\n```
# \n$JSON_BLOB\n```
# \nObservation: action result
# \n... (repeat Thought/Action/Observation N times)
# \nThought: I know what to respond\nAction:
# \n```
# \n{{{{
    # \n  "action": "Final Answer",
    # \n  "action_input": "Final response to human"
    # \n}}}}
    # \n```'

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """
You are a helpful assistant that uses tools to answer the user's queries. Always respond with a JSON object that specifies the 
action to take. Use the following format:

{
    "action": "ToolName",
    "action_input": "Input for the tool"
}

If the answer can be provided without using any tools, use the following format:
{
    "action": "Final Answer",
    "action_input": "Your final answer to the user."
}

Available tools: {tools}
Begin!
"""

human_template = """
User: {input}
Chat History: {agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", human_template),
])

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Define the tools the agent can use
tools = [tool_1, tool_2]

# Create the StructuredChatAgent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True  # Ensures the agent stops at the defined stop token
)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)



import json

# Define the input
input_data = {
    "input": "What is 15 multiplied by 7?"
}

# Invoke the agent
response = agent_executor.invoke(input_data)

# Print the structured output
print(json.dumps(response, indent=4))
> LangChain BigTool
!pip install langgraph-bigtool

from langchain_ollama import ChatOllama


import math
import types
import uuid

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import (
    convert_positional_only_function_to_tool
)

# Collect functions from `math` built-in
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(
        function, types.BuiltinFunctionType
    ):
        continue
    # This is an idiosyncrasy of the `math` library
    if tool := convert_positional_only_function_to_tool(
        function
    ):
        all_tools.append(tool)

# Create registry of tools. This is a dict mapping
# identifiers to tool instances.
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# Index tool names and descriptions in the LangGraph
# Store. Here we use a simple in-memory store.
embeddings = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {
            "description": f"{tool.name}: {tool.description}",
        },
    )

# Initialize agent
llm = init_chat_model("openai:gpt-4o-mini")

# # qwen2.5 - 14b (#30 on BFCL leaderboard)
# local_llm = "qwen2.5:14b"
# llm = ChatOllama(model=local_llm, temperature=0.0)


builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)
agent





# ------------------------------------- Customizing Tool Retrieval --------------------------------
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from typing_extensions import Annotated


def retrieve_tools(
    query: str,
    # Add custom arguments here...
    *,
    store: Annotated[BaseStore, InjectedStore],
) -> list[str]:
    """Retrieve a tool to use, given a search query."""
    results = store.search(("tools",), query=query, limit=2)
    tool_ids = [result.key for result in results]
    # Insert your custom logic here...
    return tool_ids

builder = create_agent(
    llm, tool_registry, retrieve_tools_function=retrieve_tools
)
agent = builder.compile(store=store)
> CrewAI Agents (Advanced Collaboration)
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_community.llms import OpenAI

# Define agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge technologies and market trends',
    backstory="You are an experienced technology analyst with a knack for identifying emerging trends.",
    max_iter=5,  # Limit reasoning steps
    llm=ChatOpenAI(model="gpt-4-turbo", temperature=0.3),
    verbose=True,
    memory=True,  # Maintains conversation history
    tools=[SerperDevTool()],  # Search tool
    allow_delegation=True
)

# # Usage
# research_result = research_agent.execute(
#     "Find recent breakthroughs in AI-driven drug discovery"
# )

writer = Agent(
    role='Tech Content Strategist',
    goal='Create compelling content about technology trends',
    backstory="You transform complex technical concepts into engaging content.",
    llm=OpenAI(temperature=0.7),
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research emerging AI technologies focusing on practical applications in healthcare",
    expected_output="A comprehensive report on emerging AI in healthcare with at least 5 specific technologies",
    agent=researcher
)

writing_task = Task(
    description="Create an engaging blog post based on the research findings",
    expected_output="A 1000-word blog post with sections covering each major technology",
    agent=writer,
    context=[research_task]
)

# Create the crew
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=2
)

# Execute the crew
result = tech_crew.kickoff()
> AutoGen Conversational Agents (Microsoft)
from autogen import ConversableAgent, GroupChatManager

# Create specialized agents
data_scientist = ConversableAgent(
    name="Data_Scientist",
    system_message="Expert in statistical analysis and ML modeling",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

domain_expert = ConversableAgent(
    name="Medical_Expert",
    system_message="Healthcare domain expert with clinical trial experience",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

# Advanced group chat manager
group_chat_manager = GroupChatManager(
    groupchat_participants=[data_scientist, domain_expert],
    max_round=10,
    admin_name="Moderator"
)
> Google Vertex AI Agents
from vertexai.preview.generative_models import GenerativeModel, Part, Tool, ToolConfig, ToolUseBlock
import vertexai

# Initialize Vertex AI
vertexai.init(project="your-project-id", location="us-central1")

# Define tools
def get_weather(location: str) -> str:
    """Gets the current weather for a given location."""
    # This would typically call a weather API
    return f"Sunny, 72°F in {location}"

tools = [
    Tool(
        function_declarations=[
            {
                "name": "get_weather",
                "description": "Gets the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for, e.g. 'San Francisco, CA'"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    )
]

# Create the model with tool configuration
model = GenerativeModel(
    "gemini-pro",
    tools=tools,
    generation_config={"temperature": 0.2}
)

# Handle function execution
def handle_tool_call(tool_call):
    if tool_call.name == "get_weather":
        location = tool_call.args["location"]
        response = get_weather(location)
        return response
    return "Unknown tool"

# Generate content with tool use
response = model.generate_content(
    "What's the weather like in Seattle right now?",
)

# Process any tool calls in the response
if hasattr(response, 'candidates') and len(response.candidates) > 0:
    for part in response.candidates[0].content.parts:
        if isinstance(part, ToolUseBlock):
            # Process and respond to the tool call
            tool_result = handle_tool_call(part.function_call)
            
            # Continue the conversation with the tool result
            follow_up = model.generate_content(
                [
                    Part.from_text("What's the weather like in Seattle right now?"),
                    response.candidates[0].content,
                    Part.from_function_response(
                        name=part.function_call.name,
                        response=tool_result
                    )
                ]
            )
            print(follow_up.text)
> PandasAI Data Agent
from pandasai import SmartDataFrame
from pandasai.llm import OpenAI

# Initialize advanced data agent
llm = OpenAI(api_token="sk-...", model="gpt-4")
df = SmartDataFrame(
    "medical_data.csv",
    config={
        "llm": llm,
        "enable_cache": False,
        "max_retries": 5,
        "custom_prompts": {
            "clean_data": "Automatically clean and preprocess this dataset"
        }
    }
)

# Execute complex analysis
response = df.chat(
    "Predict which drug candidates have >80% efficacy probability "
    "using Bayesian regression analysis"
)
> Create Custom Agent
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
import re
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

class CustomAgent(BaseModel):
    llm: BaseLanguageModel  # The LLM to use for decision-making
    tools: List[BaseTool]  # List of tools the agent can use
    max_loops: int = 5  # Maximum number of loops to prevent infinite execution
    stop_pattern: List[str]  # Stop patterns for the LLM to avoid hallucinations

    @property
    def tool_by_names(self) -> Dict[str, BaseTool]:
        """Map tool names to tool objects."""
        return {tool.name: tool for tool in self.tools}

    def run(self, question: str) -> str:
        """Run the agent to answer a question."""
        name_to_tool_map = self.tool_by_names
        previous_responses = []
        num_loops = 0

        while num_loops < self.max_loops:
            num_loops += 1

            # Format the prompt with the current state
            curr_prompt = PROMPT_TEMPLATE.format(
                tool_description="\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                tool_names=", ".join([tool.name for tool in self.tools]),
                question=question,
                previous_responses="\n".join(previous_responses),
            )

            # Get the next action from the LLM
            output, tool, tool_input = self._get_next_action(curr_prompt)

            # If the final answer is found, return it
            if tool == "Final Answer":
                return tool_input

            # Execute the tool and get the result
            tool_result = name_to_tool_map[tool].run(tool_input)
            output += f"\n{OBSERVATION_TOKEN} {tool_result}\n{THOUGHT_TOKEN}"
            print(output)  # Print the agent's reasoning
            previous_responses.append(output)

        return "Max loops reached without finding a final answer."

    def _get_next_action(self, prompt: str) -> Tuple[str, str, str]:
        """Get the next action from the LLM."""
        result = self.llm.generate([prompt], stop=self.stop_pattern)
        output = result.generations[0][0].text  # Get the first generation

        # Parse the output to extract the tool and input
        tool, tool_input = self._get_tool_and_input(output)
        return output, tool, tool_input

    def _get_tool_and_input(self, generated: str) -> Tuple[str, str]:
        """Parse the LLM output to extract the tool and input."""
        if FINAL_ANSWER_TOKEN in generated:
            return "Final Answer", generated.split(FINAL_ANSWER_TOKEN)[-1].strip()

        # Use regex to extract the tool and input
        regex = r"Action: (.*?)\nAction Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        if not match:
            raise ValueError(f"Output of LLM is not parsable for next tool use: `{generated}`")

        tool = match.group(1).strip()
        tool_input = match.group(2).strip(" ").strip('"')
        return tool, tool_input
    

FINAL_ANSWER_TOKEN = "Final Answer:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PROMPT_TEMPLATE = """Answer the question as best as you can using the following tools: 

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: comment on what you want to do next
Action: the action to take, exactly one element of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation repeats N times, use it until you are sure of the answer)
Thought: I now know the final answer
Final Answer: your final answer to the original input question

Begin!

Question: {question}
Thought: {previous_responses}
"""

# The tool(s) that your Agent will use
tools = [annual_report_tool]

# The question that you will ask your Agent
question = "What was Meta's net income in 2022? What was net income the year before that?"

# The prompt that your Agent will use and update as it is "reasoning"
prompt = PROMPT_TEMPLATE.format(
  tool_description="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
  tool_names=", ".join([tool.name for tool in tools]),
  question=question,
  previous_responses='{previous_responses}',
)

# The LLM that your Agent will use
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

# Initialize your Agent
agent = CustomAgent(
  llm=llm, 
  tools=tools, 
  prompt=prompt, 
  stop_pattern=[f'\n{OBSERVATION_TOKEN}', f'\n\t{OBSERVATION_TOKEN}'],
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=False,  # Only final output. If True, returns all intermediate steps
    handle_parsing_errors=True,  # Graceful parsing errors
)
# Run the Agent!
result = agent.run(question)

print(result)
> Use Langchain "initialize_agent" class
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


tools = [tool_1,tool_2]  # Add more tools as needed

# Define the agent's prompt
prompt_template = """
You are an advanced agent with access to multiple tools. Your task is to resolve customer queries by:
1. Identifying the problem or request.
2. Using the tools provided to gather additional information if needed.
3. Synthesizing the information into a clear, concise response.

You can chain tools if required. If you are unsure, respond with 'I need more details.'

Query: {query}
"""
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    streaming=True,
    callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
)

# Initialize the agent
advanced_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"prompt_template": prompt_template},
    verbose=True,
)

agent_executor = AgentExecutor(agent=advanced_agent, tools=tools, verbose=True, 
                            return_intermediate_steps=True, handle_parsing_errors=True)

agent_executor.invoke({"query": "What is the capital of France?"})
## LangGraph
# 1. Key Concepts
    # Graph : A workflow of nodes and edges.
    # Nodes : Functions or agents that perform tasks.
    # Edges : Connections between nodes that define the flow.
    # State : A shared data structure passed between nodes.
    # StateGraph : A graph that manages state transitions.

# Draw a directory tree for the src directory for a LangChain project.
"""
src/
├── agents/
│   ├── __init__.py
│   ├── agent.py
│   ├── graph.py
│   ├── tools.py
│   ├── configuration.py
│   ├── state.py
│   ├── prompts.py
│   └── utils.py

"""
> LangGraph Workflow
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from typing import Literal, Sequence, List, Annotated
from typing_extensions import TypedDict
import functools
import operator

pip install --upgrade --force-reinstall langgraph

#------------------ Define the Memory Saver-----------------------
memory = MemorySaver()

#------------------ Define the State-----------------------     # You can write a custom state class by extending the TypedDict class.
class AgentState(TypedDict):
    # Messages: Stores conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Selected Agents: Tracks which agents are active in the workflow
    selected_analysts: List[str]
    
    # Current Agent Index: Tracks the progress through the selected agents
    current_analyst_idx: int

workflow = StateGraph(AgentState)   # Initialize the Graph

#------------------ Create Nodes-----------------------
def supervisor_router(state):
    """Route to appropriate analyst(s) based on the query"""
    result = routing_chain.invoke(state)
    selected_analysts = [a.strip() for a in result.content.strip().split(',')]
    return {
        "messages": state["messages"] + [SystemMessage(content=f"Routing query to: {', '.join(selected_analysts)}", name="supervisor")],
        "selected_analysts": selected_analysts,
        "current_analyst_idx": 0
    }

# or
def agent_node(state: AgentState, agent, name: str) -> AgentState:
    """
    Generic node function for an agent.
    - `state`: The current state of the workflow.
    - `agent`: The agent or function to process the state.
    - `name`: The name of the agent (for logging or identification).
    """
    # Invoke the agent with the current state
    result = agent.invoke(state)
    
    # Update the state with the agent's output
    return {
        "messages": state["messages"] + [HumanMessage(content=result["messages"][-1].content, name=name)],
        "selected_agents": state["selected_agents"],
        "current_agent_idx": state["current_agent_idx"] + 1
    }


#--------------------- Wrap the agent in a node--------------------------

# Create the analysts with their specific tools
quant_strategist = create_react_agent(llm, tools=quant_strategist_tools)
quant_strategist_node = functools.partial(agent_node, agent=quant_strategist, name="quant_strategist")

macro_analyst = create_react_agent(llm, tools=macro_analyst_tools)
macro_analyst_node = functools.partial(agent_node, agent=macro_analyst, name="macro_analyst")


#------------------- Add Nodes to Graph-----------------------
workflow = StateGraph(AgentState)   # Initialize the Graph
workflow.add_node("supervisor", supervisor_router)  # Add the supervisor node
workflow.add_node("quant_strategist", quant_strategist_node)    # Add the quant_strategist node
workflow.add_node("macro_analyst", macro_analyst_node)        # Add the macro_analyst node

#------------------- Define the Prompt-----------------------
class SupervisorPrompt(ChatPromptTemplate):
    """Prompt for the supervisor node"""
    messages: MessagesPlaceholder
    selected_analysts: List[str]
    current_analyst_idx: int

#------------------- Define Conditional Edge-----------------------
def get_next_step(state: AgentState) -> str:
    """
    Determines the next step in the workflow.
    - If no agents are selected, go to the final summary.
    - If all agents have processed, go to the final summary.
    - Otherwise, go to the next agent.
    """
    if not state["selected_agents"]:
        return "final_summary"
    current_idx = state["current_agent_idx"]
    if current_idx >= len(state["selected_agents"]):
        return "final_summary"
    return state["selected_agents"][current_idx]


# Add conditional edges:
workflow.add_conditional_edges(
    "supervisor",  # Source node
    get_next_step,  # Router node/Function to determine the next step
    {
        "quant_strategist": "quant_strategist",  # Route to quant_strategist node
        "macro_analyst": "macro_analyst",        # Route to macro_analyst node
        "final_summary": "final_summary"         # Route to final_summary node
    }
)

#------------------ Add Final Edges ------------------------------------
workflow.add_edge(START, "supervisor")
workflow.add_edge("final_summary", END)

#-------------------- Compile the Graph --------------------------------
graph = workflow.compile()
# or
graph = workflow.compile(checkpointer=memory)   # Compile the graph with memory
# or
graph = workflow.compile(checkpointer=memory, interrupt_before=["quant_strategist_node"])  # Compile the graph with memory and interrupt before quant_strategist_node


#------------------ Stream the Graph with Memory--------------------------------
config = {"configurable": {"thread_id": "1"}}   # add memory thread, we used thread_id = 2
events = graph.stream({"messages": {"Hi there, my name is Paul"}}, config, stream_mode = "values")

for event in events:    # Iterate over the events
    event['messages'][-1].pretty_print()

memory.get(config)  # Retrieve the memory for a specific configuration or thread_id


#-------------------- Accessing the Graph State --------------------------------
graph = workflow.compile()
graph.get_state(config).values  # get the state of the graph
graph.get_state(config).values.get("messages", "")  # get the messages from the state
graph.update_state(config, {"input": "Hello, World!"})  # update the state of the graph
> Nice way to execute the LangGraph
#---------------------------ATLERNATIVE WAY TO RUN THE GRAPH IN A BEAUTIFUL WAY------------------------------



#------------------------- Run the Graph------------------------------------
#------------------------- Custom Function----------------------------------
from typing import Dict, Any
import json
import re
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

#---------- Formatting Functions
# Format Bold Text
def format_bold_text(content: str) -> Text:
    """Convert **text** to rich Text with bold formatting."""
    text = Text()
    pattern = r'\*\*(.*?)\*\*'
    parts = re.split(pattern, content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text.append(part)
        else:
            text.append(part, style="bold")
    return text

# Format Message Content
def format_message_content(content: str) -> Union[str, Text]:
    """Format the message content, handling JSON and text with bold markers."""
    try:
        data = json.loads(content)
        return json.dumps(data, indent=2)
    except:
        if '**' in content:
            return format_bold_text(content)
        return content

# Format Agent Message
def format_agent_message(message: HumanMessage) -> Union[str, Text]:
    """Format a single agent message."""
    return format_message_content(message.content)

# Get Agent Title
def get_agent_title(agent: str, message: HumanMessage) -> str:
    """Get the title for the agent panel, with fallback handling."""
    base_title = agent.replace('_', ' ').title()
    if hasattr(message, 'name') and message.name is not None:
        try:
            return message.name.replace('_', ' ').title()
        except:
            return base_title
    return base_title

# Print a Single Step
def print_step(step: Dict[str, Any]) -> None:
    """Pretty print a single step of the agent execution."""
    console = Console()
    for agent, data in step.items():
        # Handle supervisor steps
        if 'next' in data:
            next_agent = data['next']
            text = Text()
            text.append("Portfolio Manager ", style="bold magenta")
            text.append("assigns next task to ", style="white")
            if next_agent == "final_summary":
                text.append("FINAL SUMMARY", style="bold yellow")
            elif next_agent == "END":
                text.append("END", style="bold red")
            else:
                text.append(f"{next_agent}", style="bold green")
            console.print(Panel(
                text,
                title="[bold blue]Supervision Step",
                border_style="blue"
            ))
        # Handle agent responses and final summary
        if 'messages' in data:
            message = data['messages'][0]
            formatted_content = format_agent_message(message)
            if agent == "final_summary":
                # Final summary formatting
                console.print(Rule(style="yellow", title="Portfolio Analysis"))
                console.print(Panel(
                    formatted_content,
                    title="[bold yellow]Investment Summary and Recommendation",
                    border_style="yellow",
                    padding=(1, 2)
                ))
                console.print(Rule(style="yellow"))
            else:
                # Regular analyst reports
                title = get_agent_title(agent, message)
                console.print(Panel(
                    formatted_content,
                    title=f"[bold blue]{title} Report",
                    border_style="green"
                ))

# Stream the Execution
def stream_agent_execution(graph, input_data: Dict, config: Dict) -> None:
    """Stream and pretty print the agent execution."""
    console = Console()
    console.print("\n[bold blue]Starting Agent Execution...[/bold blue]\n")
    for step in graph.stream(input_data, config):
        if "__end__" not in step:
            print_step(step)
            console.print("\n")
    console.print("[bold blue]Analysis Complete[/bold blue]\n")


# Run the Graph
# Define the input data
input_data = {
    "messages": [HumanMessage(content="What is AAPL's current price and latest revenue?")]
}

# Define the configuration (e.g., recursion limit)
config = {"recursion_limit": 10}

# Stream the execution
stream_agent_execution(graph, input_data, config)
> LangGraph States
# LangGraph State: --> Example
# What is a LangGraph State?
    # A LangGraph state is a data structure that holds the current state of the workflow. It is passed between nodes in the graph, 
    # and each node can modify the state as needed. The state typically contains all the information required for the workflow to function, 
    # such as inputs, intermediate results, and outputs.

from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
import operator
from langgraph.graph import Graph, StateGraph, MessageGraph, MessagesState

 
#------------------ Define the State (State.py) -----------------------
DEFAULT_EXTRACTION_SCHEMA = {
    "title": "CompanyInfo",
    "description": "Basic information about a company",
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Official name of the company",
        },
        "founding_year": {
            "type": "integer",
            "description": "Year the company was founded",
        },
        "founder_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the founding team members",
        },
        "product_description": {
            "type": "string",
            "description": "Brief description of the company's main product or service",
        },
        "funding_summary": {
            "type": "string",
            "description": "Summary of the company's funding history",
        },
    },
    "required": ["company_name"],
}

class SampleState(MessagesState):   # this state will have both company and messages (since messages is already defined in the MessagesState)
    """A sample state class that extends the MessagesState."""
    company: str
    """Company to research provided by the user."""
    

@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    # Messages: Stores conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    # Messages: Stores conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"

    
@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

#------------------ Define the Configuration (Configuration.py) -----------------------
@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""

    max_search_queries: int = 3  # Max search queries per company
    max_search_results: int = 3  # Max search results per query
    max_reflection_steps: int = 0  # Max reflection steps

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
#-------------------------------------------------------------------------------
from langgraph.graph import START, END, StateGraph
from agent.configuration import Configuration

builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
> LangGraph Nodes
# DAG: Directed Acyclic Graph 
    # Definition : A graph where nodes are connected in a linear, directional manner without forming closed loops .
    # Use Case : Used by LangChain to represent workflows where tasks are executed in a non-repeating, linear sequence .
        ''' Start → Node A → Node B → Node C → End '''
            # No loops : Once a node is processed, it doesn’t revisit previous nodes.
            # Linear flow : Tasks are executed in a strict sequence.
            
            
# DCG: Directed Cyclic Graph --> used by LangGraph to represent the workflow of nodes and edges.
    # Definition : A graph where nodes are connected in a directional manner and can form loops or cycles .
    # Use Case : Used by LangGraph to represent workflows with complex patterns , including loops and conditional branching .
        '''
        Start → Node A → Node B → Node C
                ↑              ↓
                └──────────────┘
        '''
            # Loops allowed : Nodes can revisit previous nodes (e.g., for iterative tasks).
            # Complex flow : Supports conditional edges, loops, and dynamic routing.


# Edges:
    # Simple Edge:
        # A direct connection between two nodes in the graph. Used whrn the flow is fixed and uncontitional.
        ''' Start → Node A → Node B → Node C → End '''
    
    # Conditional Edge:
        # A connection between two nodes that is determined by a condition or decision function.
        '''
            Start → Node A
                    ↓
                ┌─────┴─────┐
            Condition 1   Condition 2
                ↓             ↓
            Node B         Node C
                ↓             ↓
            Node D         Node E
                └─────┬─────┘
                    ↓
                    End
        '''
from langgraph.graph import START, END, StateGraph, Graph # Import the necessary classes
from IPython.display import Image, display
from pydantic import BaseModel, Field
from IPython.display import Image, display

# Define the state as a Pydantic model
class CustomerSupportState(BaseModel):
    query: str = Field(..., description="The customer's query")
    response: str = Field(None, description="The response to the customer")
    issue_type: str = Field(None, description="The type of issue (FAQ, Escalation, Recommendation)")
    escalation_required: bool = Field(False, description="Whether the issue requires escalation")
    product_recommendation: str = Field(None, description="Product recommendation for the customer")

# Create the workflow graph
workflow = StateGraph(CustomerSupportState)


#--------------------------------------------- NODE WITHOUT LLM ---------------------------------------------
#------------------------------------------------------------------------------------------------------------
# Node A: Classify the customer's query
def classify_query(state: CustomerSupportState) -> dict:
    query = state.query.lower()
    if "faq" in query or "how to" in query or "what is" in query:
        return {"issue_type": "FAQ"}
    elif "issue" in query or "problem" in query or "error" in query:
        return {"issue_type": "Escalation"}
    elif "recommend" in query or "suggest" in query:
        return {"issue_type": "Recommendation"}
    else:
        return {"issue_type": "Unknown"}

#--------------------------------------------- TOOL NODE ---------------------------------------------
#-----------------------------------------------------------------------------------------------------
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode

vectorstore=Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chrome",
    embedding=embeddings
    
)
retriever=vectorstore.as_retriever()
retriever_tool=create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.You are a specialized assistant. Use the 'retriever_tool' **only** when the query explicitly relates to LangChain blog data. For all other queries, respond directly without using any tool. For simple queries like 'hi', 'hello', or 'how are you', provide a normal response.",
    )

tools=[retriever_tool]
retrieve=ToolNode([retriever_tool])



#--------------------------------------------- TOOL NODE WITH FALLBACK ---------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")
    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls
    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # Format the error message for the user
                tool_call_id=tc["id"],  # Associate the error message with the corresponding tool call ID
            )
            for tc in tool_calls  # Iterate over each tool call to produce individual error messages
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    Args:
        tools (list): A list of tools to be included in the node.
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],  # Use a lambda function to wrap the error handler
        exception_key="error"  # Specify that this fallback is for handling errors
    )

builder = StateGraph(OverallState)
builder.add_node("tools", create_tool_node_with_fallback([retriever_tool, tool_1, tool_2]))


#--------------------------------------------- NODE with LLM 1 ---------------------------------------------
#-----------------------------------------------------------------------------------------------------------
from langgraph.graph import add_messages
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def ai_assistant(state:AgentState):
    print("---CALL AGENT---")
    messages = state['messages']
    
    if len(messages)>1:
        last_message = messages[-1]
        question = last_message.content
        prompt=PromptTemplate(
        template="""You are a helpful assistant whatever question has been asked to find out that in the given question and answer.
                        Here is the question:{question}
                        """,
                        input_variables=["question"]
                        )
            
        chain = prompt | llm
    
        response=chain.invoke({"question": question})
        return {"messages": [response]}
    else:
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        #response=handle_query(messages)
        return {"messages": [response]}



#--------------------------------------------- NODE with LLM + structured output ---------------------------------------------
#-----------------------------------------------------------------------------------------------------------
class grade(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")
    
def grade_documents(state:AgentState)->Literal["Output_Generator", "Query_Rewriter"]:
    llm_with_structure_op=llm.with_structured_output(grade)
    
    prompt=PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
                    Here is the document: {context}
                    Here is the user’s question: {question}
                    If the document talks about or contains information related to the user’s question, mark it as relevant. 
                    Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
        input_variables=["context", "question"]
                    )
    chain = prompt | llm_with_structure_op
    
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generator" #this should be a node name
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewriter" #this should be a node name


#--------------------------------------------- NODE with LLM + structured output 2 ----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
query_writer_instructions="""Your goal is to generate targeted web search query.

The query will gather information related to a specific topic.

Topic:
{research_topic}

Return your query as a JSON object:
{{
    "query": "string",
    "aspect": "string",
    "rationale": "string"
}}
"""

def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """
    
    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )   
    query = json.loads(result.content)
    
    return {"search_query": query['query']}



#------------------------------------------------ NODE with LLM + structured output 3 ----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

RECOMMENDATION_PROMPT_2 = """
You are a specialized travel recommendation assistant. 
Generate at least 10 unique recommendations for the user. 
Output your response as a list of dictionaries, each containing the fields: 
  - "key": A short label, e.g. "Crime Rate"
  - "value": A concise recommendation

For example:
[
    {"key": "Crime Rate", "value": "The city is generally safe but beware of pickpockets in tourist areas."},
    {"key": "Weather Advice", "value": "Spring is mild; pack light jackets and an umbrella."}
]

### User Query:
{query}
"""

def recommendations_node_2(state: OverallState) -> OverallState:
    import openai
    
    # If you haven't set up your API key globally, do so here:
    # openai.api_key = "YOUR_OPENAI_API_KEY"
    
    client = openai  # or adapt to your environment if needed

    # Combine all user messages into a single query
    all_messages = "\n".join([message.content for message in state.messages])
    preferences_text = "\n".join([f"{key}: {value}" for key, value in state.user_preferences.items()])
    query = f"{all_messages}\n\nUser Preferences:\n{preferences_text}"

    # Define the structured response format with a JSON Schema
    completion = client.chat.completions.create(
        model="gpt-4o",  # Replace with a valid model you have access to.
        messages=[
            {"role": "system", "content": RECOMMENDATION_PROMPT_2},
            {"role": "user", "content": query},
        ],
        # The 'response_format' parameter needs 'json_schema' -> 'name' + 'schema'
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "recommendation_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "recommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {
                                        "type": "string",
                                        "description": "Short label for the recommendation"
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "Concise recommendation content"
                                    }
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False
                            },
                            "description": "A list of travel recommendations."
                        }
                    },
                    "required": ["recommendations"],
                    "additionalProperties": False
                }
            },
        },
    )

    # Parse and return the generated structured output
    try:
        structured_output = completion.choices[0].message.content
        parsed_output = json.loads(structured_output)
        recommendation_list = parsed_output["recommendations"]  # This is the list of dictionaries
        transformed_list = [{item["key"]: item["value"]} for item in recommendation_list]
        
        state.recommendations = transformed_list
        
        return state 
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"error": "Failed to generate recommendations."}



#--------------------------------------------- NODE with LLM + Blind Tools (Used for User call) ---------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
# from langchain_aws import ChatBedrock
import boto3
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

@tool
def compute_savings(monthly_cost: float) -> float:
    """
    Tool to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.
    
    Args:
        monthly_cost (float): The user's current monthly electricity cost.
    
    Returns:
        dict: A dictionary containing:
            - 'number_of_panels': The estimated number of solar panels required.
            - 'installation_cost': The estimated installation cost.
            - 'net_savings_10_years': The net savings over 10 years after installation costs.
    """
    def calculate_solar_savings(monthly_cost):
        # Assumptions for the calculation
        cost_per_kWh = 0.28  
        cost_per_watt = 1.50  
        sunlight_hours_per_day = 3.5  
        panel_wattage = 350  
        system_lifetime_years = 10  
        # Monthly electricity consumption in kWh
        monthly_consumption_kWh = monthly_cost / cost_per_kWh
        
        # Required system size in kW
        daily_energy_production = monthly_consumption_kWh / 30
        system_size_kW = daily_energy_production / sunlight_hours_per_day
        
        # Number of panels and installation cost
        number_of_panels = system_size_kW * 1000 / panel_wattage
        installation_cost = system_size_kW * 1000 * cost_per_watt
        
        # Annual and net savings
        annual_savings = monthly_cost * 12
        total_savings_10_years = annual_savings * system_lifetime_years
        net_savings = total_savings_10_years - installation_cost
        
        return {
            "number_of_panels": round(number_of_panels),
            "installation_cost": round(installation_cost, 2),
            "net_savings_10_years": round(net_savings, 2)
        }
    # Return calculated solar savings
    return calculate_solar_savings(monthly_cost)

# Define the state for the workflow
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the assistant class used for invoking the runnable
class Assistant:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable
    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break
        # Return the final state after processing the runnable
        return {"messages": result}


llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        streaming=True,
        callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    )

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful customer support assistant for Solar Panels Belgium.
            You should get the following information from them:
            - monthly electricity cost
            If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
            After you are able to discern all the information, call the relevant tool.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)

# Define the tools the assistant will use
part_1_tools = [
    compute_savings
]

# Bind the tools to the assistant's workflow
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools, tool_choice="any")

builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))




#--------------------------------------- NODE with LLM + Tools with multiple parameters ----------------------------
#-------------------------------------------------------------------------------------------------------------------


def accommodation_finder_node(state: OverallState) -> OverallState:
    """
    This node extracts accommodation details from the user's query in state.messages
    and returns a structured output that can be passed to the booking tool.
    """

    class AccommodationOutput(BaseModel):
        location: str = Field(..., description="The exact location or neighborhood where the traveler wants to stay (e.g., 'Brooklyn').")
        checkin_date: str = Field(..., description="The check-in date in YYYY-MM-DD format.")
        checkout_date: str = Field(..., description="The check-out date in YYYY-MM-DD format.")
        adults: int = Field(default=2, description="The number of adult guests.")
        rooms: int = Field(default=1, description="The number of rooms.")
        currency: str = Field(default="USD", description="The currency for the prices.")
        
    # Create a new LLM with structured output
    llm_with_structure = llm.with_structured_output(AccommodationOutput)

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        You are an advanced travel planner assistant. Your task is to extract accommodation details
        from the traveler's query. Use the following information to generate a structured output for
        booking accommodations:

        ### Traveler Query:
        {query}

        ### Instructions:
        1. Extract the exact location or neighborhood where the traveler wants to stay (e.g., "Brooklyn").
           - If the traveler does not specify a location, use the city or city code provided in the state.
        2. Extract the check-in and check-out dates from the query.
           - If the dates are not explicitly mentioned, use the default dates from the state.
        3. Extract the number of adults and rooms from the query.
           - If not specified, use the default values: 1 adult and 1 room.
        4. Use the default currency 'USD' unless specified otherwise.
        5. Return the structured output in the following format:
           - location: The exact location or neighborhood.
           - checkin_date: The check-in date in YYYY-MM-DD format.
           - checkout_date: The check-out date in YYYY-MM-DD format.
           - adults: The number of adult guests.
           - rooms: The number of rooms.
           - currency: The currency for the prices.

        ### Example Output:
        - location: "Brooklyn"
        - checkin_date: "2023-12-01"
        - checkout_date: "2023-12-10"
        - adults: 2
        - rooms: 1
        - currency: "USD"
        """,
        input_variables=["query"]
    )

    # Create the chain
    chain = prompt | llm_with_structure

    # Extract the user's query from state.messages
    query = state.messages[-1].content  # Assuming the last message is the user's query

    # Invoke the chain to generate the structured output
    structured_output = chain.invoke({"query": query})

    # Call Google Flights Search Tool        
    booking_search_input = BookingSearchInput(
        location=structured_output.location,
        checkin_date=structured_output.checkin_date,
        checkout_date=structured_output.checkout_date,
        adults=structured_output.adults,
        rooms=structured_output.rooms,
        currency=structured_output.currency,
    )

    booking_results = booking_tool.func(booking_search_input)
    
    # Update the state with the structured output
    state.accommodation = booking_results

    # Return the updated state
    return state



#--------------------------------------------- NODE with Agent 1 ---------------------------------------------
#-------------------------------------------------------------------------------------------------------------
from langchain.agents import Tool, create_react_agent

# Define a REACT-based agent node
def react_agent_node(state: CustomerSupportState):
    tools = [retriever_tool]  # Add your tool(s) here
    prompt_template = """You are a reasoning and acting agent.
    Use the tools available to gather or verify information as needed.
    Respond directly if no tools are required.

    Question: {query}
    """
    react_agent = create_react_agent(
        tools=tools,
        prompt_template=prompt_template,
        llm=llm,
    )
    
    agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
    # Execute the REACT agent
    query = state.query
    response = agent_executor.invoke({"query": query})
    state.response = response
    return state


#--------------------------------------------- NODE with Agent 2 ---------------------------------------------
#-------------------------------------------------------------------------------------------------------------
import functools

def agent_node(state: OverallState, agent, name: str) -> OverallState:
    """
    Generic node function for an agent.
    - `state`: The current state of the workflow.
    - `agent`: The agent or function to process the state.
    - `name`: The name of the agent (for logging or identification).
    """
    # Invoke the agent with the current state
    result = agent.invoke(state)
    
    # Update the state with the agent's output
    return {
        "messages": state["messages"] + [HumanMessage(content=result["messages"][-1].content, name=name)],
        "selected_agents": state["selected_agents"],
        "current_agent_idx": state["current_agent_idx"] + 1
    }

# wrap the agent in a node
def query_param_generator_node(agent_node):

    query_param_generator_agent = create_react_agent(llm, tools=[retriever_tool], prompt=prompt)
    query_param_node = functools.partial(agent_node, agent=query_param_generator_agent, name="query_param_generator")
    return query_param_node



#--------------------------------------------- NODE with Agent 3 ---------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# more advanced node

from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType, initialize_agent, AgentExecutor


def advanced_multi_tool_agent_node(state: CustomerSupportState):
    """
    An advanced agent that uses multiple tools to handle queries.
    """
    tools = [retriever_tool]  # Add more tools as needed

    # Define the agent's prompt
    prompt_template = """
    You are an advanced agent with access to multiple tools. Your task is to resolve customer queries by:
    1. Identifying the problem or request.
    2. Using the tools provided to gather additional information if needed.
    3. Synthesizing the information into a clear, concise response.

    You can chain tools if required. If you are unsure, respond with 'I need more details.'

    Query: {query}
    """
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        streaming=True,
        callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    )
    
    # Initialize the agent
    advanced_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prompt_template": prompt_template},
        verbose=True,
    )

    agent_executor = AgentExecutor(agent=advanced_agent, tools=[tool_1, tool_2], verbose=True, 
                               return_intermediate_steps=True, handle_parsing_errors=True)

    # Execute the agent
    query = state.query
    try:
        response = agent_executor.invoke({"query": query})
        state.response = response
    except Exception as e:
        state.response = f"Error: {str(e)}"
    return state




#--------------------------------------------- NODE with Custom Agent --------------------------------------
#-----------------------------------------------------------------------------------------------------------

from langchain.agents import BaseAgent
from typing import Optional

class AdvancedCustomAgent(BaseAgent):
    """
    Custom advanced agent with LLM, human-in-the-loop, and iterative reasoning.
    """
    def __init__(self, llm, tools=None, max_iterations: int = 3):
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations

    async def run(self, query: str, human_review: bool = False, **kwargs) -> str:
        """
        Executes the custom agent's workflow.
        
        Args:
            query (str): User's query.
            human_review (bool): If True, adds human-in-the-loop for review.
        
        Returns:
            str: Final response.
        """
        response = ""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration}/{self.max_iterations} ---")

            # Generate a response using LLM
            prompt = f"""
            You are an advanced customer support agent. Use the tools provided to solve the query. 
            Tools: {', '.join([tool.name for tool in self.tools]) if self.tools else 'None'}

            Query: {query}

            If you need clarification or further details, request them from the user.
            """
            try:
                response = await self.llm.apredict(prompt)
                print(f"Generated Response: {response}")

                # Check if human review is required
                if human_review:
                    review = input("Do you approve this response? (yes/no): ")
                    if review.lower() == "yes":
                        break
                    else:
                        query = input("Provide additional details or corrections: ")
                else:
                    break

            except Exception as e:
                response = f"Error: {str(e)}"
                break

        return response


# Define the custom agent node
async def custom_agent_node(state: CustomerSupportState):
    """
    Node with a custom advanced agent that uses LLM and human-in-the-loop.
    """
    custom_agent = AdvancedCustomAgent(llm=llm, tools=[retriever_tool], max_iterations=3)
    query = state.query

    # Human-in-the-loop enabled for critical queries
    response = await custom_agent.run(query, human_review=True)
    state.response = response
    return state


#--------------------------------------------- Supervisor NODE with Router + Command ---------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Agent Supervisor Node
from typing import Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState
from langgraph.types import Command

'''
User
 ├── Supervisor
 │     ├── [direct] Agent 1
 │     ├── [conditional] Agent 2 (if condition A is met)
 │     └── [direct] Agent 3
 │           ├── [conditional] Sub-Agent 3.1 (if condition B is met)
 │           └── [direct] Sub-Agent 3.2
 │
 └── Feedback Loop (User <--> Supervisor)

'''
members = ["researcher", "coder"]

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]

    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    return supervisor_node

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_edge(START, "supervisor")


#--------------------------------------------- More Advanced Supervisor NODE -------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")

math_agent = create_react_agent(
    model=model,
    tools=[tool_1, tool_2],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[tool_1, tool_2],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    output_mode = "last_message",    # what we pass back from agent to supervisor. we dont need to always state this
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})


#--------------------------------------------- Supervisor managing other Supervisors------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

math_supervisor = create_supervisor(
    [research_agent, math_agent],
    model=model,
    output_mode = "last_message",    # what we pass back from agent to supervisor. we dont need to always state this
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
).compile(name="math_supervisor")


research_supervisor = create_supervisor(
    [research_agent, math_agent],
    model=model,
    output_mode = "last_message",    # what we pass back from agent to supervisor. we dont need to always state this
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
).compile(name="research_supervisor")

workflow = create_supervisor(
    [math_supervisor, research_supervisor],
    supervisor_name = "top_level_supervisor",
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
).compile()




#--------------------------------------------- NODE with Command (used with tool node) ---------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
llm = llm.bind_tools([retriever_tool, tool_1, tool_2])

def call_model(state:OverallState) -> Command[Literal['tools', END]]:
    message = state.messages[-1].content
    response = llm.invoke(message)
    if len(response.tool_calls) > 0:
        next_node = "tools"
    else:
        next_node = END
    return Command(goto=next_node, update={"messages": response})   # update helps to update the state with the response from the model. You dont need always need to do this

workflow = StateGraph(OverallState)
workflow.add_node("call_model", call_model)
workflow.add_node("tools", create_tool_node_with_fallback([retriever_tool, tool_1, tool_2]))
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", "tools")
graph = workflow.compile()


# If you are using subgraphs, you might want to navigate from a node a subgraph to a different subgraph 
# (i.e. a different node in the parent graph).
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )

#--------------------------------------------- LangGraph Workflow ---------------------------------------------   
#-----------------------------------------------------------------------------------------------------------
# Add nodes to the workflow
workflow.add_node("Classify Query", classify_query)
workflow.add_node("End Conversation", end_conversation)

# Define edges between nodes
workflow.add_edge(START, "Classify Query")
workflow.add_edge("Classify Query", "Answer FAQ")
workflow.add_edge("Recommend Products", "End Conversation")
workflow.add_edge("End Conversation", END)

# Set entry and finish points
workflow.set_entry_point("Classify Query")  # Start the conversation. Use this only when START is not used.
workflow.set_finish_point("End Conversation")   # End the conversation. Use this only when END is not used.

# Compile the workflow
app = workflow.compile()

# Test the workflow with a sample query
initial_state = CustomerSupportState(query="which do you recommend between product A and product B?")
result = app.invoke(initial_state)

> langgraph-swarm
# Installation and Setup
pip install langgraph-swarm langchain-openai
export OPENAI_API_KEY=<your_api_key>  # Or use a .env file


# Core Components and Code
    from langgraph_swarm import create_handoff_tool, create_swarm, add_active_agent_router
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph_swarm.swarm import SwarmState
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    
    # langgraph_swarm:  The main library.
    # create_handoff_tool(agent_name: str, description: str | None = None) -> BaseTool:  
        # This is the crucial function. It creates a LangChain Tool that, when invoked by an agent, triggers a handoff to another agent.
            # agent_name: The name of the agent to transfer control to (must match the node name in the graph).
            # description: A description of when this tool should be used. This is important for the LLM to understand the tool's purpose.

        from langgraph_swarm import create_handoff_tool
        from langgraph.prebuilt import create_react_agent

        transfer_to_bob = create_handoff_tool(
            agent_name="Bob",
            description="Ask Bob for help with pirate speak."
        )

        alice = create_react_agent(
            model=ChatOpenAI(model="gpt-4"),
            tools=[some_tool, transfer_to_bob],
            prompt="You are Alice, an expert at math. If you can't help, hand off to Bob.",
            name="Alice",
        )


    # create_swarm(agents: list[CompiledStateGraph], *, default_active_agent: str, state_schema: StateSchemaType = SwarmState) -> StateGraph: 
        # This function builds the overall multi-agent graph (the "swarm").
            # agents: A list of pre-compiled LangGraph agents (each is a CompiledStateGraph). These are your individual, specialized agents.
            # default_active_agent: The agent that starts the conversation.
            # state_schema: (Optional, but important) Defines the structure of the state that's passed between agents and within the swarm. 
                # The default SwarmState includes messages (conversation history) and active_agent (who's currently in control).

            workflow = create_swarm(
                agents=[agent1, agent2, ...],
                default_active_agent="Alice",  # or whichever agent to start with
                state_schema=MySwarmState      # By default uses SwarmState
            )

    # add_active_agent_router(builder: StateGraph, *, route_to: list[str], default_active_agent: str) -> StateGraph: 
        # Adds routing logic to the graph to switch between agents based on the active_agent in the state.
            # builder: The StateGraph instance being built.
            # route_to: A list of valid agent names (node names) that can be routed to.
            # default_active_agent: The agent to start with if no agent is active.

    # InMemorySaver():  This is a checkpointer. It's responsible for storing and retrieving the conversation state (including messages and 
        # active_agent).  InMemorySaver keeps it in memory (good for testing, not for production).  LangGraph also supports other checkpointers 
        # (e.g., Redis, databases).

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        checkpointer = InMemorySaver()  # short-term memory
        store = InMemoryStore()         # optional long-term store

        app = swarm_workflow.compile(
            checkpointer=checkpointer,
            store=store
        )

    # SwarmState:  A TypedDict (from typing_extensions) that defines the structure of the state.  By default, it includes:
        # messages: Annotated[list[AnyMessage], add_messages] - The conversation history, using LangChain's AnyMessage type. The add_messages 
            # annotation is important for LangGraph to know how to update this list.
        # active_agent:str The current active agent.
        from langgraph_swarm.swarm import SwarmState

        # Minimal example: SwarmState is a typed dictionary that includes
        # "messages" and "active_agent" by default.
        class MySwarmState(SwarmState):
            # Optionally, you can add any other keys here if needed
            pass


# --------------------------------------------- Customizing Handoff Tools ----------------------------------------------------
from typing import Annotated
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command

def create_custom_handoff_tool(agent_name: str) -> BaseTool:
    @tool(name="custom_transfer", description=f"Custom Transfer to {agent_name}")
    def handoff(
        extra_context: Annotated[str, "Additional context for the next agent."],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_msg = ToolMessage(
            content=f"Transferred with extra context: {extra_context}",
            name="custom_transfer",
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": state["messages"] + [tool_msg],
                "active_agent": agent_name,
                "some_extra_field": extra_context,
            },
        )
    return handoff

import datetime
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

model = ChatOpenAI(model="gpt-4o")

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
import datetime
from collections import defaultdict

# --- Mock Data (Replace with real data/APIs) ---
reservations = defaultdict(lambda: {"flights": [], "hotels": []})
tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

flights_data = [
  {"id": "F123", "from": "BOS", "to": "JFK", "date": tomorrow, "airline": "JetBlue"},
  {"id": "F456", "from": "LAX", "to": "SFO", "date": tomorrow, "airline": "United"},
]

hotels_data = [
  {"id": "H123", "city": "New York", "name": "The Plaza", "stars": 5},
  {"id": "H456", "city": "Los Angeles", "name": "The Beverly Hills Hotel", "stars": 5},
]

# --- Flight Agent Tools ---
def search_flights(departure_airport: str, arrival_airport: str, date: str) -> list[dict]:
    """Searches for flights based on departure, arrival, and date."""
    # In a real application, this would query a database or API.
    results = []
    for flight in flights_data:
        if (flight["from"] == departure_airport and
            flight["to"] == arrival_airport and
            flight["date"] == date):
            results.append(flight)
    return results

def book_flight(flight_id: str, config:dict) -> str:
    """Books a flight given its ID."""
    user_id = config["configurable"].get("user_id")
    for flight in flights_data:
      if flight["id"] == flight_id:
          reservations[user_id]["flights"].append(flight)
          return f"Booked flight {flight_id}."
    return f"Could not book flight {flight_id}."

# --- Hotel Agent Tools ---
def search_hotels(city: str) -> list[dict]:
    """Searches for hotels in a given city."""
    # In a real application, this would query a database or API.
    results = []
    for hotel in hotels_data:
      if hotel["city"] == city:
          results.append(hotel)
    return results

def book_hotel(hotel_id: str, config: dict) -> str:
    """Books a hotel given its ID."""
    user_id = config["configurable"].get("user_id")
    for hotel in hotels_data:
        if hotel["id"] == hotel_id:
            reservations[user_id]["hotels"].append(hotel)
            return f"Booked hotel {hotel_id}."
    return f"Could not book hotel {hotel_id}."

# --- Create Handoff Tools ---
transfer_to_hotel = create_handoff_tool(
    agent_name="HotelAgent",
    description="Transfer to the hotel booking assistant for help with finding and booking hotels.",
)

transfer_to_flight = create_handoff_tool(
    agent_name="FlightAgent",
    description="Transfer to the flight booking assistant for help with finding and booking flights.",
)

# --- Create Agents ---
llm = ChatOpenAI(model="gpt-4o")  # Or any other suitable model

flight_agent = create_react_agent(
    llm,
    [search_flights, book_flight, transfer_to_hotel],
    prompt="You are a helpful flight booking assistant. Help users find and book flights.",
    name="FlightAgent",
)

hotel_agent = create_react_agent(
    llm,
    [search_hotels, book_hotel, transfer_to_flight],
    prompt="You are a helpful hotel booking assistant. Help users find and book hotels.",
    name="HotelAgent",
)

# --- Create Swarm ---
checkpointer = InMemorySaver()
workflow = create_swarm(
    [flight_agent, hotel_agent],
    default_active_agent="FlightAgent"
)

app = workflow.compile(checkpointer=checkpointer)




# ------------------------------------ Run the swarm ------------------------------------
# --- Example Interaction ---
config = {"configurable": {"thread_id": "user123", "user_id": "user123"}}  # Use a consistent thread ID!
result = app.invoke({"messages": [{"role": "user", "content": "I need a flight from Boston to NYC tomorrow."}]}, config)
print(result)

result = app.invoke({"messages": [{"role": "user", "content": "And a 5-star hotel."}]}, config) # Keep the same config
print(result)

result = app.invoke({"messages": [{"role": "user", "content": "What's my reservation?"}]}, config) # Keep the same config
print(result)

> LangGraph CodeAct
pip install langgraph-codeact
# also install langchain-openai
pip install langchain langchain-anthropic


# --------------------------------------------- Tools Setup ------------------------------------------------

# Define your tools (functions the LLM can use)
from langchain_core.tools import tool

@tool
def search_database(query: str) -> list:
    """Search the database with the given query."""
    # Implementation
    return results

# Create a list of all tools
tools = [search_database, other_tool, ...]



# --------------------------------------------- Setting Up the CodeAct Graph ------------------------------------------------
from langchain.chat_models import init_chat_model
from langgraph_codeact import create_codeact, create_default_prompt
from langgraph.checkpoint.memory import MemorySaver

# Initialize LLM
model = init_chat_model("claude-3-7-sonnet-latest", model_provider="anthropic")

# Create CodeAct instance
code_act = create_codeact(model, tools, eval)

# code_act = create_codeact(
#     model,
#     tools,
#     eval,
#     prompt=create_default_prompt(
#         tools,
#         "Once you have the final answer, respond to the user with plain text, do not respond with a code snippet.",
#     ),
# )

# Compile with checkpointer to maintain state between interactions
agent = code_act.compile(checkpointer=MemorySaver())



# ---------------------------------------------- Running the CodeAct Graph ------------------------------------------------
# For final output only
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your query here"}]
})

# For streaming output
for typ, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Your query here"}]},
    stream_mode=["values", "messages"],
    config={"configurable": {"thread_id": 1}}
):
    if typ == "messages":
        print(chunk[0].content, end="")
    elif typ == "values":
        print("\nFinal result:", chunk)


# ------------------------------------------ Code Sandbox Implementation ------------------------------------------------

# Basic sandbox function (DO MOT USE IN PRODUCTION)
    # DO NOT USE THIS IN PRODUCTION - EXAMPLE ONLY
    def eval(code: str, _locals: dict) -> tuple[str, dict]:
        # This is NOT safe for production use
        import io, contextlib
        original_keys = set(_locals.keys())
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(code, {}, _locals)
            result = f.getvalue() or "<code ran, no output>"
        except Exception as e:
            result = f"Error: {repr(e)}"
        
        # Track new variables created during execution
        new_keys = set(_locals.keys()) - original_keys
        new_vars = {key: _locals[key] for key in new_keys}
        return result, new_vars



# RestrictedPython
from RestrictedPython import compile_restricted, safe_globals

def secure_sandbox(code: str, _locals: dict) -> tuple[str, dict]:
    try:
        byte_code = compile_restricted(code, filename="<inline>", mode="exec")
        exec_globals = safe_globals.copy()
        exec_globals.update({
            "_getattr_": getattr,
            "_write_": lambda x: None,  # Replace with output capture
            "__builtins__": {"__import__": lambda name: None}  # Block imports
        })
        
        # Add your tools to the globals
        for tool_name, tool_func in _locals.items():
            if callable(tool_func):
                exec_globals[tool_name] = tool_func
                
        # Execute the code
        exec(byte_code, exec_globals)
        
        # Capture new variables
        new_vars = {k: v for k, v in exec_globals.items() 
                   if k not in safe_globals and not k.startswith('_')}
        
        return "Code executed successfully", new_vars
    except Exception as e:
        return f"Error: {str(e)}", {}
    
    


# Docker-based Sandbox
import docker
import uuid
import tempfile
import os

def docker_sandbox(code: str, _locals: dict) -> tuple[str, dict]:
    client = docker.from_env()
    
    # Create a temporary file with the code
    run_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    code_file = os.path.join(temp_dir, f"code_{run_id}.py")
    
    # Serialize tools and variables to be imported in the container
    tool_code = "# Tool definitions\n"
    for name, func in _locals.items():
        if callable(func):
            tool_code += f"# Tool: {name}\n"
    
    with open(code_file, 'w') as f:
        f.write(tool_code + "\n" + code)
    
    # Run in container with strict limitations
    try:
        container = client.containers.run(
            "python:3.10-slim",
            command=f"python /code/code_{run_id}.py",
            volumes={temp_dir: {'bind': '/code', 'mode': 'ro'}},
            mem_limit="50m",
            cpu_quota=10000,  # 10% of CPU
            network_mode="none",  # No network access
            detach=True
        )
        
        # Wait for execution with timeout
        result = container.wait(timeout=5)
        output = container.logs().decode('utf-8')
        container.remove()
        
        # Placeholder for returning variables (needs additional implementation)
        # In practice, you would need to serialize output variables from container
        return output, {}
    
    except Exception as e:
        return f"Error: {str(e)}", {}
    finally:
        # Clean up
        os.remove(code_file)
        os.rmdir(temp_dir)



# PyPy Sandbox
from pypy.interpreter.gateway import unwrap_spec
from pypy.translator.sandbox.sandlib import SandboxedProc

def pypy_sandbox(code: str, _locals: dict) -> tuple[str, dict]:
    # Implementation details would depend on PyPy setup
    # This is a simplified skeleton
    
    proc = SandboxedProc(['pypy'])
    # Setup sandbox with limited resources and capabilities
    
    # Execute the code
    try:
        result = proc.interact(code)
        # Handle capturing new variables
        return result, {}
    except Exception as e:
        return f"Error: {str(e)}", {}
    

# Pysandbox (Using separate process)
import subprocess
import json
import tempfile
import os

def process_sandbox(code: str, _locals: dict) -> tuple[str, dict]:
    # Create a file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
        # Prepare tool definitions to be available
        tools_code = "# Tool definitions\n"
        for name, func in _locals.items():
            if callable(func):
                tools_code += f"# {name} is available as a tool\n"
        
        # Write the code to the file
        temp.write(tools_code + "\n" + code)
        temp_name = temp.name
    
    try:
        # Execute in a separate process with resource limits
        result = subprocess.run(
            ["python", temp_name],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nErrors: {result.stderr}"
            
        # In a real implementation, you'd need to serialize variables
        # from the subprocess back to the main process
        return output, {}
    
    except subprocess.TimeoutExpired:
        return "Execution timed out", {}
    except Exception as e:
        return f"Error: {str(e)}", {}
    finally:
        os.unlink(temp_name)



> TrustCall
# TrustCall solves LLM JSON generation problems by using JSON patch operations instead of full generation:
    # ⚡ Faster & cheaper structured output generation
    # 🐺 Resilient validation error retrying (works with pydantic, schema dictionaries, or Python functions)
    # 🧩 Accurate schema updates without losing information
    # Works for extraction, routing, and multi-step agent workflows

pip install trustcall   # installation


# Basic Usage
    # 1. Schema Definition with Pydantic

        from typing import List, Optional
        from pydantic import BaseModel, Field

        class Address(BaseModel):
            street: str
            city: str
            state: str
            zip_code: str

        class Contact(BaseModel):
            email: str
            phone: Optional[str] = None

        class Product(BaseModel):
            id: str
            name: str
            description: str
            price: float
            category: str
            in_stock: bool
            tags: List[str] = Field(default_factory=list)

        class Customer(BaseModel):
            customer_id: str
            first_name: str
            last_name: str
            address: Address
            contact: Contact
            purchase_history: List[Product] = Field(default_factory=list)
            loyalty_points: int = 0
            notes: Optional[str] = None
    
    
    # 2. Creating an Extractor

        from langchain_openai import ChatOpenAI
        from trustcall import create_extractor

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o")

        # Create extractor with defined schema
        extractor = create_extractor(
            llm, 
            tools=[Customer], 
            tool_choice="Customer"
        )
        

    # 3. Extracting Data
        # Extract customer data from text
        conversation = """
        Rep: Thank you for calling. May I have your name?
        Customer: John Smith, ID CS-4921.
        Rep: Thanks Mr. Smith. What can I help you with?
        Customer: I need to update my address. I've moved to 123 Oak Street, Portland, Oregon 97205.
        Rep: Got it. And is your contact information still johsmith@email.com and 555-867-5309?
        Customer: The email is correct, but my phone is now 555-123-4567.
        Rep: Updated. I see you've purchased our Premium Router last month. How is it working?
        Customer: It's great! Much faster than my old one.
        Rep: Wonderful. For your loyalty, I've added 500 points to your account. Anything else?
        Customer: That's all, thanks!
        """

        result = extractor.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract customer information from this conversation:\n{conversation}"
                }
            ]
        })

        print(result["responses"][0])


    # 4. Updating Existing Schema
        # Initial customer data
        existing_customer = {
            "customer_id": "CS-4921",
            "first_name": "John",
            "last_name": "Smith",
            "address": {
                "street": "456 Pine Ave",
                "city": "Seattle",
                "state": "WA",
                "zip_code": "98101"
            },
            "contact": {
                "email": "johnsmith@email.com",
                "phone": "555-867-5309"
            },
            "purchase_history": [
                {
                    "id": "PR-789",
                    "name": "Basic Router",
                    "description": "Entry-level wireless router",
                    "price": 49.99,
                    "category": "Networking",
                    "in_stock": True,
                    "tags": ["wifi", "budget"]
                }
            ],
            "loyalty_points": 100,
            "notes": "Prefers email communication"
        }

        # Update with new information
        result = extractor.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Update the customer record with new information from this conversation:\n{conversation}"
                }
            ],
            "existing": {"Customer": existing_customer}
        })

        print(result["responses"][0])


    # 5. Simultaneous Updates & Insertions
        from typing import Dict, List

        # Multiple customer records
        customers = [
            Customer(
                customer_id="CS-4921",
                first_name="John",
                last_name="Smith",
                address=Address(street="456 Pine Ave", city="Seattle", state="WA", zip_code="98101"),
                contact=Contact(email="johnsmith@email.com", phone="555-867-5309"),
                purchase_history=[],
                loyalty_points=100
            ),
            Customer(
                customer_id="CS-3856",
                first_name="Sarah",
                last_name="Johnson",
                address=Address(street="789 Maple Dr", city="Chicago", state="IL", zip_code="60601"),
                contact=Contact(email="sjohnson@email.com", phone="555-234-5678"),
                purchase_history=[],
                loyalty_points=250
            )
        ]

        # Convert to format expected by extractor
        existing_data = [
            (str(i), "Customer", customer.model_dump()) for i, customer in enumerate(customers)
        ]

        # Create extractor that allows insertions
        extractor = create_extractor(
            llm,
            tools=[Customer],
            tool_choice="any",
            enable_inserts=True
        )

        # Update existing records and add new ones
        conversation = """
        Rep: I have notes from two customers today. 
        First, John Smith called to update his address to 123 Oak Street, Portland, OR 97205.
        His new phone number is 555-123-4567.
        Second, we have a new customer Jane Doe, ID CS-5678, who just purchased our Premium Router.
        Her address is 555 Elm Court, Austin, TX 78701. 
        Email is janedoe@email.com, phone 555-987-6543.
        She seems very tech-savvy, so note that down.
        """

        result = extractor.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Update existing customer records and add new ones based on this conversation:\n{conversation}"
                }
            ],
            "existing": existing_data
        })

        # Print the results
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            print(f"ID: {rmeta.get('json_doc_id', 'New')}")
            print(r.model_dump_json(indent=2))
            print()







> LangGraph Graph Call
from langgraph.errors import GraphRecursionError
from langchain.schema import HumanMessage, SystemMessage, AIMessage  # Import AIMessage for assistant responses

#----------------------------------------------------- Graoh Invoke-----------------------------------

# **Input Collection**
user_input = "I want to travel from New York to Paris on 2023-12-15 and return on 2023-12-22. There are 2 adults and 1 child. My budget is $5000. I need 1 room. I prefer a hotel with free breakfast and a swimming pool. I also want to visit the museums and enjoy local cuisine, and go to the club at night. I might also want a massage."    

# **Input State**
input_state = {"messages": [HumanMessage(content=user_input)]}

# **Graph Invocation**
output = graph.invoke(input_state, {"recursion_limit": 300})



#--------------------------------------------------------- Graph Stream-----------------------------------
# Stream the Graph
# Define the input data
input_data = {
    "messages": [HumanMessage(content="I want to travel from New York to Paris on 2023-12-15 and return on 2023-12-22. There are 2 adults and 1 child. My budget is $5000. I need 1 room. I prefer a hotel with free breakfast and a swimming pool. I also want to visit the museums and enjoy local cuisine, and go to the club at night. I might also want a massage.")]
}

# Define the configuration (e.g., recursion limit)
config = {"recursion_limit": 10}

# Stream the execution
events = graph.stream(input_data, config)
for event in events:
    print(event)
    
#--------------------------------------------------------- Graph Stream with Memory-----------------------------------


thread_config = {"configurable": {"thread_id": "1"}}

try:
    for event in graph.stream(input_state, thread_config, stream_mode = "values", ):
        messages = event['messages'][-1]
        # Filter and print only the AIMessage content
        if isinstance(messages, AIMessage):
            print(messages.content)

except GraphRecursionError:
    print("Recursion Error")


#--------------------------------------------------------- Graph Stream with pretty print-----------------------------------

from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage, SystemMessage

graph_name = "task_maistro" 

# Connect to the deployment
remote_graph = RemoteGraph(graph_name, url=local_deployment_url)

user_input = "Hi I'm Lance. I live in San Francisco with my wife and have a 1 year old."
config = {"configurable": {"user_id": "Test-Deployment-User"}}
for chunk in remote_graph.stream({"messages": [HumanMessage(content=user_input)]}, stream_mode="values", config=config):
    convert_to_messages(chunk["messages"])[-1].pretty_print()
    
    
> Langchain Tool Call
#-------------------------------------------------- Tool with Input Class and Tool Class ----------------------------------
#--------------------------------------------------------------------------------------------------------------------------
# Define the input class
class MyToolInput(BaseModel):
    param1: str
    param2: int

# Define the tool class
class MyTool:
    def __call__(self, input: MyToolInput) -> str:
        # Tool logic here
        return f"Processed: {input.param1}, {input.param2}"

my_tool = Tool(
    name="my_tool",
    func=MyTool(),
    description="Tool description.",
    args_schema=MyToolInput
)

# Call the tool
result = my_tool.func(MyToolInput(param1="value1", param2=42))
print(result)



#-------------------------------------------------- Tool Using @tool Decorator ----------------------------------
#----------------------------------------------------------------------------------------------------------------
from langchain.tools import tool

@tool
def my_tool(param1: str, param2: int) -> str:
    """Tool description."""
    return f"Processed: {param1}, {param2}"

# Call the tool
result = my_tool({"param1": "value1", "param2": 42})
print(result)



#-------------------------------- Tool with Structured Inputs Using "BaseTool" ----------------------------------
#----------------------------------------------------------------------------------------------------------------
from langchain.tools import BaseTool
from pydantic import BaseModel

class MyToolInput(BaseModel):
    param1: str
    param2: int

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Tool description."

    def _run(self, param1: str, param2: int) -> str:
        """Tool logic."""
        return f"Processed: {param1}, {param2}"

# Create an instance of the tool
my_tool = MyTool()

# Call the tool
result = my_tool.run({"param1": "value1", "param2": 42})
print(result)



#--------------------------------------------------  Tool call with Agent ----------------------------------
#-----------------------------------------------------------------------------------------------------------
from langchain.agents import initialize_agent, Tool

def my_tool_func(param1: str, param2: int) -> str:
    """Tool logic."""
    return f"Processed: {param1}, {param2}"

# Create the tool
my_tool = Tool(
    name="my_tool",
    func=my_tool_func,
    description="Tool description."
)

# Add the tool to an agent
tools = [my_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")   # you can also use react agent here or any agent

# Call the tool via the agent
result = agent.invoke("Call my_tool with param1='value1' and param2=42")
print(result)
> Langchain Prompt Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

#--------------------------------------------- Style 1 ---------------------------------------------
#---------------------------------------------------------------------------------------------------

# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.
"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_gen_system),
        ("placeholder", "{messages}"),
    ]
)


#--------------------------------------------- Style 2 ---------------------------------------------
#---------------------------------------------------------------------------------------------------
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful customer support assistant for Solar Panels Belgium.
            You should get the following information from them:
            - monthly electricity cost
            If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
            After you are able to discern all the information, call the relevant tool.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)


#--------------------------------------------- Style 3 ---------------------------------------------
#---------------------------------------------------------------------------------------------------
template = ''' 
You are a travel suggestion agent. Answer the user's questions based on their travel preferences. 
If you need to find information about a specific destination, use the search_tool. Understand that the information was retrieved from the web,
interpret it, and generate a response accordingly.

Answer the following questions as best as you can. You have access to the following tools:
{tools}

Use the following format:

"Question": the input question you must answer
"Thought": your reasoning about what to do next
"Action": the action you should take, one of [{tool_names}] (if no action is needed, write "None")
"Action Input": the input to the action (if no action is needed, write "None")
"Observation": the result of the action (if no action is needed, write "None")
"Thought": your reasoning after observing the action
"Final Answer": the final answer to the original input question

Ensure every Thought is followed by an Action, Action Input, and Observation. If no tool is needed, explicitly write "None" for Action, Action Input, and Observation.

Begin!
Question: {input}
Thought: {agent_scratchpad}
'''

prompt = PromptTemplate.from_template(template)

agent_executor = AgentExecutor(
    agent=search_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=False,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({
    "input": latest_query,
    "agent_scratchpad": ""  # Initialize with an empty scratchpad
})
> Display or Visualize LangGraph
#--------------------------------------------------------
print(app.get_graph().draw_mermaid())       # Converting a Graph to a Mermaid Diagram


#-------------------------Using Mermaid.Ink--------------------------------
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


#-------------------------Using Mermaid + Pyppeteer--------------------------------
import nest_asyncio

nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

display(
    Image(
        app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.API,
            background_color="white",
            padding=10,
        )
    )
)


#-------------------------Using Graphviz--------------------------------
%pip install pygraphviz

display(Image(app.get_graph().draw_png()))
> LangGraph Deployment
# Build the graph
agent = workflow.compile()


# requirements.txt
langgraph==0.1.0
langchain_core==0.1.0


# Langgraph.json
{
  "name": "todo_agent",
  "description": "A simple ToDo list agent",
  "graphs": {
    "todo_agent": {
      "entrypoint": "agent",
      "file": "agent.py"
    }
  }
}


# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["langgraph", "serve", "--host", "0.0.0.0", "--port", "8123"]



# or
# Use docker-compose.yml to create containers for Redis, PostgreSQL, and the LangGraph API.
$ cd module-6/deployment
$ docker compose up

# Building Docker Image
$ langgraph build -t todo_agent

# Run the Docker Image
docker run -p 8123:8123 todo_agent
#-------------------------Deployment Setup--------------------------------
# Use docker-compose.yml to create containers for Redis, PostgreSQL, and the LangGraph API.
$ cd module-6/deployment
$ docker compose up

# Building Docker Image
$ langgraph build -t todo_agent

# Run the Docker Image
docker run -p 8123:8123 todo_agent

#-------------------------Assistants-------------------------------------
#---------------------
# Creating Assistants (Connect to the Deployment)
#---------------------
from langgraph_sdk import get_client
client = get_client(url="http://localhost:8123")

# Create a personal assistant
personal_assistant = await client.assistants.create(
    "task_maistro",
    config={"configurable": {"todo_category": "personal"}}
)

#---------------------
# Updating Assistants
#---------------------
personal_assistant = await client.assistants.update(
    personal_assistant["assistant_id"],
    config={"configurable": {"todo_category": "personal", "user_id": "lance"}}
)

#---------------------
# Searching and Deleting Assistants
#---------------------
assistants = await client.assistants.search()
for assistant in assistants:
    print(assistant['assistant_id'], assistant['config'])
    

await client.assistants.delete("assistant_id")  # Delete an assistant


#-------------------------Threads and Runs--------------------------------

#---------------------
# Creating Threads
#---------------------
thread = await client.threads.create()


#---------------------
# Running a Graph
#---------------------
run = await client.runs.create(
    thread["thread_id"],
    "task_maistro",
    input={"messages": [HumanMessage(content="Add a ToDo")]},
    config={"configurable": {"user_id": "Test"}}
)

#---------------------
# Streaming Runs
#---------------------
async for chunk in client.runs.stream(
    thread["thread_id"],
    "task_maistro",
    input={"messages": [HumanMessage(content="What ToDo should I focus on?")]},
    stream_mode="messages-tuple"
):
    if chunk.event == "messages":
        print(chunk.data)



#---------------------
# Background Runs
#---------------------
run = await client.runs.create(thread["thread_id"], "task_maistro", input={"messages": [...]})
print(await client.runs.get(thread["thread_id"], run["run_id"]))


#-------------------------Double Texting Strategies--------------------------------

#---------------------
# Reject
#---------------------
await client.runs.create(
    thread["thread_id"],
    "task_maistro",
    input={"messages": [HumanMessage(content="New ToDo")]},
    multitask_strategy="reject" # Reject the current task if another task is already in progress ()
)


#---------------------
# Enqueue
#---------------------
await client.runs.create(
    thread["thread_id"],
    "task_maistro",
    input={"messages": [HumanMessage(content="New ToDo")]},
    multitask_strategy="enqueue"    # Enqueue new runs (or Interrupt, Rollback etc.)
)


#-------------------------Human-in-the-Loop--------------------------------

#---------------------
# Forking Threads
#---------------------
copied_thread = await client.threads.copy(thread["thread_id"])

#---------------------
# Editing State
#---------------------
forked_input = {"messages": HumanMessage(content="Updated ToDo", id=message_id)}
await client.threads.update_state(
    thread["thread_id"],
    forked_input,
    checkpoint_id=checkpoint_id
)



> LangGraph App Deployment
# use this template toolkit:
https://github.com/JoshuaC215/agent-service-toolkit     #shows how to deploy agent service in FastAPi and Azure, plus streamlit for chat
# and
https://github.com/onyx-dot-app/onyx        # better chat app interface for LangGraph
https://github.com/vercel/ai-chatbot?tab=readme-ov-file # Vercel AI Chatbot - a simple chat app template

# Here is a walkthrough video
https://www.youtube.com/watch?v=pdYVHw_YCNY



# Sample Streamlit Apps built with Langgraph and Langchain
https://github.com/langchain-ai/streamlit-agent

https://github.com/lucasboscatti/sales-ai-agent-langgraph   # A Virtual Sales Agent that uses LangChain, LangGraph,



https://blog.langchain.dev/how-we-deployed-our-multi-agent-flow-to-langgraph-cloud-2/ # Deploy LangGraph app in LangGraph CLoud:




# Agentic RAG + more
https://github.com/SciPhi-AI/R2R #R2R
https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22 #RAG with Hybrid Search and Reranking
https://github.com/athina-ai/rag-cookbooks #Agentic RAG (Very Good)
https://github.com/NovaSearch-Team/RAG-Retrieval  # Unify Efficient Fine-tuning of RAG Retrieval, including Embedding, ColBERT, ReRanker.
https://www.analyticsvidhya.com/blog/2025/04/advanced-rag-techniques/ #RAG entire workflow
https://github.com/AnswerDotAI/RAGatouille  # Colbert for RAG


# RAG IMPLEMENTATION
    #1. Vector Database or Document Store
    #2. Embedding Model - OpenAI or HuggingFace ('BAAI/bge-base-en-v1.5')
    #3. LLM (Language Model) - Zephyr-7b-alpha (HuggingFace)
    #4. Use a ReRanker (rerank with Cohere-Rerank)
    #5. Retriever:
        # a. Vector Search (e.g., FAISS, Pinecone, Weaviate)
        # b. Document Store Search (e.g., Elasticsearch, Weaviate)
        # c. Hybrid Search with ensemble Retrieval
        # d. Using GLiNER to Generate Metadata to add context to the documents
        # d. Reranking with LLMs (Re-ranking with Cohere-Rerank)
        # d. Agentic RAG with LangGraph
    #6. Use an Agentic RAG with LangGraph (LangGraph + LangChain + LangGraph + LangChain + LangGraph)
        # a. Use LangGraph to create the Graph
        # b. Use LangChain to create the LLM and Embedding Model
        # c. Use LangGraph to create the Retriever
        # d. Use LangGraph to create the Agentic RAG

> LangGraph Deployment 2 (Using FastAPi and toolkit)

deactivate
# or
conda deactivate
#  then
.\.venv\Scripts\Activate.ps1


i created the environment using uv.
Like this.

pip install uv
uv sync --frozen
# "uv sync" creates .venv automatically
source .venv/bin/activate
python src/run_service.py

# In another shell
.\.venv\Scripts\Activate.ps1
# # Comprehensive Guide to Deploying LangGraph Agents with Agent-Service-Toolkit

# ## 1. Understanding the Service Layer: Why FastAPI?

# The `src/service` directory contains a FastAPI implementation that serves as an API interface for your LangGraph agents.
# 
# ### Why FastAPI Instead of Direct Deployment?
# - **HTTP Interface**: LangGraph agents are Python objects. To make them accessible over a network, we need an HTTP server.
# - **Streaming Support**: FastAPI allows streaming responses, crucial for real-time agent outputs.
# - **Concurrency**: FastAPI's async support enables multiple concurrent agent instances.
# - **API Management**: Provides standardized API endpoints, rate limiting, and authentication.
# - **Scalability**: Makes agents deployable to cloud environments that expect HTTP services.
#
# ### Key Components of the Service Layer
# Inside `src/service/service.py`, you'll find:

# Example of what the service layer looks like:
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from typing import Dict, Any, List, Optional

# Core FastAPI setup
app = FastAPI(
    title="Agent Service",
    description="A service for running agents",
    version="0.1.0",
)

class AgentInput(BaseModel):
    input: str
    options: Optional[Dict[str, Any]] = None

# Agent endpoints
@app.post("/{agent_name}/invoke")
async def invoke_agent(agent_name: str, input_data: AgentInput):
    """
    Invoke an agent with the given input.
    
    Args:
        agent_name: The name of the agent to invoke
        input_data: The input data for the agent
        
    Returns:
        The agent's response
    """
    # Gets the agent by name from registry
    # Runs the agent
    # Returns response
    pass

@app.post("/{agent_name}/stream")
async def stream_agent(agent_name: str, input_data: AgentInput):
    """
    Stream an agent's response with the given input.
    
    Args:
        agent_name: The name of the agent to invoke
        input_data: The input data for the agent
        
    Returns:
        A streaming response from the agent
    """
    # Similar to invoke but streams the response
    pass

# What src/run_service.py does
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.service.service:app", host="0.0.0.0", port=8080, reload=True)

# ## 2. Local Deployment Options

# ### A. Direct Local Deployment
# This is the basic approach to run the service locally:

# 1. Set up environment variables
# Create .env file
env_file_content = """OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4-turbo-preview
# Add other environment variables as needed
"""
with open(".env", "w") as f:
    f.write(env_file_content)

# 2. Install dependencies (run this in your terminal)
# pip install -e .  # Or use uv: uv sync --frozen

# 3. Run the service (run this in your terminal)
# python src/run_service.py

# The service will be available at http://localhost:8080. You can test it with curl:
curl_command = """
curl -X POST http://localhost:8080/chatbot/invoke \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Tell me a joke"}'
"""
print("Test command:", curl_command)

# ### B. Local Docker Deployment
# Docker deployment is more robust because:
# - Consistent environment across machines
# - No dependency conflicts
# - Easier transition to production

# Steps for local Docker deployment:

# 1. Set up environment variables - already done above

# 2. Build and start only the agent service (not Streamlit)
docker_build_command = """
docker compose build agent_service
docker compose up agent_service
"""
print("Docker build command:", docker_build_command)

# If you want to modify the Docker setup to only build the agent service, edit the `compose.yaml`:
docker_compose_content = """
# Modified compose.yaml for agent service only
services:
  agent_service:
    build:
      context: .
      dockerfile: docker/Dockerfile.service
    ports:
      - "8080:8080"
    env_file:
      - .env
    # Development mode watcher for code changes
    develop:
      watch:
        - path: src/agents/
          action: sync+restart
          target: /app/agents/
        - path: src/service/
          action: sync+restart
          target: /app/service/
"""
with open("docker-compose.service-only.yaml", "w") as f:
    f.write(docker_compose_content)

# ## 3. Cloud Deployment

# ### A. AWS Deployment
# To deploy to AWS, you have several options. Let's focus on AWS ECS (Elastic Container Service):

# #### Step 1: Prepare your Docker image
aws_ecr_commands = """
# Build the Docker image
docker build -t agent-service -f docker/Dockerfile.service .

# Tag and push to AWS ECR (replace with your AWS account ID and region)
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker tag agent-service:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/agent-service:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/agent-service:latest
"""
print("AWS ECR Commands:", aws_ecr_commands)

# #### Step 2: Create ECS Task Definition
ecs_task_definition = """{
  "family": "agent-service",
  "networkMode": "awsvpc",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "agent-service",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/agent-service:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "hostPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your_api_key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/agent-service",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048"
}"""

with open("task-definition.json", "w") as f:
    f.write(ecs_task_definition)

register_task_command = "aws ecs register-task-definition --cli-input-json file://task-definition.json"
print("Register task command:", register_task_command)

# #### Step 3: Create ECS Service
create_service_command = """
aws ecs create-service \\
  --cluster your-cluster \\
  --service-name agent-service \\
  --task-definition agent-service:1 \\
  --desired-count 1 \\
  --launch-type FARGATE \\
  --network-configuration "awsvpcConfiguration={subnets=[subnet-abcdef12],securityGroups=[sg-abcdef12],assignPublicIp=ENABLED}"
"""
print("Create service command:", create_service_command)

# ### B. Azure Deployment
# For Azure, we can use Azure Container Instances:
azure_commands = """
# 1. Create a resource group
az group create --name agent-service-rg --location eastus

# 2. Create a container registry
az acr create --resource-group agent-service-rg --name agentserviceregistry --sku Basic

# 3. Login to the registry
az acr login --name agentserviceregistry

# 4. Build and push the container
docker build -t agentserviceregistry.azurecr.io/agent-service:latest -f docker/Dockerfile.service .
docker push agentserviceregistry.azurecr.io/agent-service:latest

# 5. Create a container instance
az container create \\
  --resource-group agent-service-rg \\
  --name agent-service \\
  --image agentserviceregistry.azurecr.io/agent-service:latest \\
  --dns-name-label agent-service \\
  --ports 8080 \\
  --environment-variables OPENAI_API_KEY=your_api_key
"""
print("Azure Commands:", azure_commands)

# ## 4. Deploying Multiple Agents to Different Cloud Providers

# ### Step 1: Create Agent-Specific Docker Images
custom_dockerfile = """
# Dockerfile.custom-agent
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./

# Optional: Remove other agents to reduce image size
RUN mkdir -p /app/tmp && \\
    cp -r /app/agents/custom_agent.py /app/tmp/ && \\
    rm -rf /app/agents/*.py && \\
    cp -r /app/tmp/* /app/agents/

# Install dependencies and start
RUN pip install . && \\
    pip install uvicorn

CMD ["python", "-m", "src.run_service"]
"""
with open("Dockerfile.custom-agent", "w") as f:
    f.write(custom_dockerfile)

# ### Step 2: Configure Agent Registry
agents_registry = """
# Modified agents.py
from langchain.agents import Tool
from langchain.chains import LLMMathChain

from agents.custom_agent import custom_agent

# Register only the custom agent
agents = {
    "custom-agent": custom_agent,
}
"""
print("Example agent registry code:")
print(agents_registry)

# ### Step 3: Deploy to Different Cloud Providers

# For multiple deployments, you might want to create a deployment script:
deploy_script = """
# deploy.py
import argparse
import subprocess

def deploy_to_aws(agent_name, region):
    # Build Docker image for specific agent
    subprocess.run(["docker", "build", "-t", f"{agent_name}-agent", "-f", f"Dockerfile.{agent_name}", "."])
    
    # Tag and push to AWS ECR
    registry = f"123456789012.dkr.ecr.{region}.amazonaws.com"
    subprocess.run(["aws", "ecr", "get-login-password", "--region", region], 
                  stdout=subprocess.PIPE, 
                  text=True)
    
    subprocess.run(["docker", "tag", 
                   f"{agent_name}-agent:latest", 
                   f"{registry}/{agent_name}-agent:latest"])
    
    subprocess.run(["docker", "push", 
                   f"{registry}/{agent_name}-agent:latest"])
    
    # Create task definition
    # ... AWS deployment steps
    
def deploy_to_azure(agent_name):
    # Build Docker image for specific agent
    subprocess.run(["docker", "build", "-t", f"{agent_name}-agent", "-f", f"Dockerfile.{agent_name}", "."])
    
    # Push to Azure Container Registry
    registry = "agentserviceregistry.azurecr.io"
    subprocess.run(["az", "acr", "login", "--name", "agentserviceregistry"])
    
    subprocess.run(["docker", "tag", 
                   f"{agent_name}-agent:latest", 
                   f"{registry}/{agent_name}-agent:latest"])
    
    subprocess.run(["docker", "push", 
                   f"{registry}/{agent_name}-agent:latest"])
    
    # Deploy container
    # ... Azure deployment steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy agents to cloud providers")
    parser.add_argument("agent", help="Agent name")
    parser.add_argument("provider", choices=["aws", "azure"], help="Cloud provider")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    
    args = parser.parse_args()
    
    if args.provider == "aws":
        deploy_to_aws(args.agent, args.region)
    elif args.provider == "azure":
        deploy_to_azure(args.agent)
"""
with open("deploy.py", "w") as f:
    f.write(deploy_script)

# Usage:
deploy_usage = "python deploy.py custom-agent aws --region us-east-1"
print("Deployment script usage:", deploy_usage)

# ## 5. Summary: Steps for Deploying Your LangGraph Agents
#
# 1. **Integrate your agent** into the framework by adding it to `src/agents/` and registering it in `agents.py`.
#
# 2. **Choose a deployment method**:
#    - Local Python deployment: Simple but requires manual terminal management
#    - Docker deployment: More robust, easier transition to production
#    - Cloud deployment: Best for production use
#
# 3. **Deploy**:
#    - For local: `python src/run_service.py`
#    - For Docker: `docker compose up agent_service`
#    - For AWS: Use ECS with the provided Docker image
#    - For Azure: Use Container Instances with the provided Docker image
#
# 4. **Test your deployment**:
curl_test = """
curl -X POST http://your-endpoint/your-agent-name/invoke \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Your input here"}'
"""
print("Test your deployment with:", curl_test)

# The key insight is that the agent-service-toolkit provides the required 
# infrastructure (FastAPI service) to expose your LangGraph agents as API endpoints, 
# making them deployable to any environment that can run containerized applications.
> LangGraph Functional API

# The Functional API is designed to simplify the creation of AI workflows by leveraging traditional programming constructs like loops 
# and conditionals, while still providing access to LangGraph's powerful features such as human-in-the-loop , persistence/memory , 
# and streaming .


# 1. Key Concepts
    # 1.1 Entrypoint
        # Definition : The starting point of a workflow. It encapsulates the workflow logic and manages the execution flow, 
        # including handling long-running tasks and interrupts.
        
        # Key Features :
            # Manages the lifecycle of the workflow.
            # Supports interrupts for human-in-the-loop interactions.
            # Can be configured with a checkpointer for persistence.

            from langgraph.func import entrypoint
            from langgraph.checkpoint.memory import MemorySaver

            @entrypoint(checkpointer=MemorySaver(), store=InMemoryStore())
            def workflow(input_data: dict, *, previous: Any = None, store: BaseStore):
                """Main workflow entry point"""
                # Workflow logic
                return result
            # Parameters:
                # checkpointer: Persistence layer (MemorySaver, RedisCheckpointer, etc.)
                # store: Long-term storage interface
                # stream_mode: Configure streaming behavior (updates/messages/custom)

    # 1.2 Task
        # Definition : A discrete unit of work, such as an API call or data processing step, that can be executed asynchronously.
        # Key Features :
            # Returns a future-like object that can be awaited for results.
            # Can be used within an entrypoint to perform specific actions.

            from langgraph.func import task

            @task
            def process_data(input_data: dict) -> dict:
                """Long-running or complex processing"""
                time.sleep(2)  # Simulate long operation
                return {"processed": True}

    # 1.3 Human-in-the-Loop
        # Definition : A workflow that pauses for human input at critical stages, allowing for review, validation, or corrections.
        # Key Features :
            # Uses the interrupt function to pause the workflow indefinitely.
            # Resumes with the Command primitive, skipping previously completed tasks.

            from langgraph.types import interrupt

            @task
            def generate_draft(topic: str) -> str:
                return f"An essay about {topic}"

            @entrypoint(checkpointer=MemorySaver())
            def document_approval_flow(content: str):
                draft = generate_draft(content).result()
                approval = interrupt({
                    "draft": draft,
                    "action": "Approve/Reject with comments",
                    "deadline": "2024-03-01"
                })
                
                if approval.get("status") == "approved":
                    return publish_draft(draft)
                else:
                    return revise_draft(draft, approval["comments"])
                
    # 1.4 Persistence/Memory
        # Definition : The ability to store and retrieve data across different interactions or workflow executions.
        # Key Features :
            # Short-term memory : Maintains conversation history or state within a single workflow execution.
            # Long-term memory : Stores user preferences or other persistent data across multiple interactions.

            # Short-term Memory (Conversation History):
            @entrypoint(checkpointer=MemorySaver())
            def chat_agent(query: str, *, previous: list = None):
                history = previous or []
                history.append(HumanMessage(content=query))
                response = llm.invoke(history)
                history.append(AIMessage(content=response))
                return entrypoint.final(value=response, save=history)

            # Long-term Memory (User Preferences):
            @entrypoint(checkpointer=MemorySaver(), store=RedisStore())
            def personalized_recommendation(user_id: str, query: str, *, store: BaseStore):
                preferences = store.get(f"user:{user_id}:preferences") or {}
                recommendations = generate_recs(query, preferences)
                store.set(f"user:{user_id}:preferences", update_prefs(preferences, query))
                return recommendations

    # 1.5 Streaming
        # Definition : Real-time updates sent to the client as the workflow progresses.
        # Key Features :
            # Supports streaming of workflow progress , LLM tokens , and custom updates .
            # Uses the stream method to send real-time data.
            
            @entrypoint(checkpointer=MemorySaver())
            def real_time_analysis(input_data: dict, writer: StreamWriter):
                writer("Starting analysis...", stream="custom")
                for chunk in data_processor.stream(input_data):
                    writer(chunk, stream="updates")
                    processed = transform_data(chunk)
                    writer(processed, stream="messages")
                return {"status": "complete"}

            # Client-side consumption
            for chunk in workflow.stream(inputs, stream_mode=["custom", "updates", "messages"]):
                handle_stream_chunk(chunk)


#-------------------------------------------------------------------------------------------------------------------

# Advanced Patterns
    # Multi-stage Approval Workflow
    @entrypoint(checkpointer=RedisCheckpointer())
    def content_creation_pipeline(topic: str):
        draft = research_topic(topic).result()
        editor_review = interrupt({"draft": draft}, role="editor")
        revised = incorporate_feedback(draft, editor_review))
        
        legal_check = legal_review(revised).result()
        if legal_check["approved"]:
            publish(revised)
        else:
            return {"status": "legal_blocked", "reasons": legal_check["reasons"]}
        
        
    # Context-Aware Processing
    @entrypoint(checkpointer=MemorySaver(), store=PostgresStore())
    def contextual_processing(user_id: str, query: str):
        context = store.get(f"user:{user_id}:context") or {}
        enhanced_query = enrich_query(query, context)
        
        result = process_query(enhanced_query).result()
        update_context(user_id, result, store)
        
        return format_response(result, context)


# Observability & Debugging
    # Enable LangSmith tracing
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "functional-api-workflows"

    @task
    def monitored_task(input_data):
        """Auto-logged task execution"""
        return process(input_data)
# Example Use Case

import os
import time
from typing import Any, Dict, List, Optional
import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter, interrupt
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
import requests

load_dotenv()

# Setup LLM and Tools
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
search_tool = TavilySearchResults()

# Custom Tools
def get_weather(location: str) -> str:
    """Retrieves the current weather for a given location."""
    api_key = os.getenv("WEATHER_API_KEY") # Replace with your weather api key.
    if not api_key:
        return "Weather API key not found. Please set WEATHER_API_KEY in .env"
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data["cod"] != "404":
            main_data = data["main"]
            weather_data = data["weather"][0]
            temperature = main_data["temp"]
            description = weather_data["description"]
            return f"Current weather in {location}: {temperature}°C, {description}"
        else:
            return "Location not found."
    except Exception as e:
        return f"Error fetching weather: {e}"

def find_activities(location: str, interests: str) -> List[str]:
    """Finds activities based on location and interests."""
    query = f"Activities in {location} related to {interests}"
    results = search_tool.run(query)
    # Handle the case where 'url' key might be missing
    processed_results = []
    for result in results:
        if isinstance(result, dict) and 'content' in result:
            processed_results.append(result['content'])
        elif isinstance(result, str):
            processed_results.append(result)
        else:
            processed_results.append("No content found.")
    return processed_results

def book_activity(activity: str, date: str, time: str) -> str:
    """Simulates booking an activity."""
    return f"Booking confirmed for {activity} on {date} at {time}."

# Define Tasks
@task
def collect_user_profile(user_message: str, *, previous: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Collects detailed user profile."""
    messages = previous or []
    messages.append({"role": "user", "content": user_message})
    response = llm.invoke(messages).content
    messages.append({"role": "assistant", "content": response})
    return messages

@task
def generate_itinerary(profile: List[Dict[str, str]], location: str) -> str:
    """Generates a dynamic itinerary."""
    interests = [msg['content'] for msg in profile if "interests" in msg['content'].lower()]
    interests = interests[0] if interests else "general interests"

    # Invoke the tools directly
    activities = find_activities(location, interests)
    weather = get_weather(location)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an adventure planner. Create a personalized itinerary."),
        ("user", f"Profile: {profile[-1]['content']}. Location: {location}. Activities: {activities}. Weather: {weather}. Generate an itinerary."),
    ])
    response = llm.invoke(prompt.format_messages(profile=profile[-1]['content'], location=location, activities=activities, weather=weather)).content
    return response

@task
def update_itinerary(itinerary: str, feedback: str) -> str:
    """Updates the itinerary based on user feedback."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an adventure planner. Update the itinerary based on feedback."),
        ("user", f"Itinerary: {itinerary}. Feedback: {feedback}. Update the itinerary."),
    ])
    response = llm.invoke(prompt.format_messages(itinerary=itinerary, feedback=feedback)).content
    return response

# Define Entrypoint
@entrypoint(checkpointer=MemorySaver())
def adventure_curator(user_message: str, writer: StreamWriter) -> Dict[str, Any]:
    """Main adventure curation workflow."""
    writer("Collecting user profile...")
    profile = collect_user_profile(user_message).result()
    print(f'User Profile: {profile}')
    location = "Paris" #example, can be gathered from user.
    writer("Generating itinerary...")
    itinerary = generate_itinerary(profile, location).result()
    writer("Itinerary generated!")
    print(f'Generated Itinerary: {itinerary}')
    approval = interrupt({
        "itinerary": itinerary,
        "action": "Do you approve this itinerary? (yes/no)",
    })
    
    if approval.lower() == "yes":
        writer("Itinerary approved!")
        return {"itinerary": itinerary, "status": "approved"}
    else:
        writer("Itinerary modified. Please provide feedback.")
        feedback = interrupt({
            "itinerary": itinerary,
            "action": "Please provide feedback to modify the itinerary.",
        })
        writer("Generating revised itinerary...")
        revised_itinerary = update_itinerary(itinerary, feedback).result()
        return {"itinerary": revised_itinerary, "status": "modified"}

# Run the Workflow
user_input = "I love hiking and historical sites. I prefer moderate activity levels and have a budget of $1000."
config = {"configurable": {"thread_id": "1"}}
for chunk in adventure_curator.stream(user_input, stream_mode=["custom"], config=config):
    print(chunk)

# Simulate User Approval
approval_input = "no, make it more hiking focused"

if approval_input.lower() != "yes":
    for chunk in adventure_curator.stream(approval_input, stream_mode=["custom"], config=config):
        print(chunk)
> LangChain MCP Adapters Cheat Sheet

# LangChain MCP Adapters library, which enables seamless integration of Anthropic Model Context Protocol (MCP) tools with LangChain and 
# LangGraph. MCP offers a standardized way for AI models and external systems to interact.


# Key Concepts:
    # Model Context Protocol (MCP): A protocol that defines how AI models can interact with tools and external systems. 
        # It promotes interoperability and standardized tool usage.
    # LangChain MCP Adapters: A library that wraps MCP tools, making them compatible with LangChain's tool interface and LangGraph agents.
    # MCP Servers: Implementations of MCP that expose tools. Servers can be written in various languages (e.g., Python, JavaScript).
    # MCP Clients: Applications that connect to MCP servers, discover available tools, and invoke them.
    

# Core Functionality:

# Installing the Library:
    pip install mcp
    pip install langchain-mcp-adapters

# MCP Servers:
    https://github.com/modelcontextprotocol/servers
    
# MCP Servers:
    # MCP Servers can be implemented using libraries like mcp.server.fastmcp (Python).
    # Servers define "tools" using decorators (e.g., @mcp.tool()).
    # Servers can use different transports (e.g., stdio, sse).
    # Example (Python - MCP Server):
    
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Math")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @mcp.tool()
    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    if __name__ == "__main__":
        mcp.run(transport="stdio")
        
# MCP Clients:
    # LangChain MCP Adapters provide client implementations to connect to MCP servers.
    # Clients can connect to single or multiple servers.
    # Clients can load MCP tools and use them within LangChain or LangGraph.
    
# Connecting to MCP Servers:
    # langchain_mcp_adapters.client.MultiServerMCPClient: Connects to multiple servers with different transports.
    # mcp.client.stdio.stdio_client: Connects to a server using standard input/output.
    # mcp.client.sse.sse_client: Connects to a server using Server-Sent Events (SSE).
    
    # Example (Python - MCP Client - Single Server):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model="gpt-4o")

    server_params = StdioServerParameters(
        command="python",
        args=["/path/to/math_server.py"],  # Update with the path to your server
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response)
        
        
    # Example (Python - MCP Client - Multiple Servers):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model="gpt-4o")

    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["/path/to/math_server.py"],  # Update with the path
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
        print(math_response)
        print(weather_response)


# Loading MCP Tools:
    # langchain_mcp_adapters.tools.load_mcp_tools: Asynchronously loads MCP tools from a ClientSession.
    # MCP tools are converted into LangChain BaseTool instances.
    # LangChain tools can then be used with LangChain agents or LangGraph workflows.

# Using MCP Tools:
    # LangChain tools created from MCP tools can be used like any other LangChain tool.
    # They can be passed to LangChain agents (e.g., create_react_agent).
    # They can be used within LangGraph nodes.


# Key Classes and Functions:
    # langchain_mcp_adapters.client.MultiServerMCPClient: Manages connections to multiple MCP servers.
        # __init__(connections: dict[str, StdioConnection | SSEConnection] = None): Initializes the client.
        # connect_to_server(server_name: str, *, transport: Literal["stdio", "sse"] = "stdio", **kwargs) -> None: Connects to a server.
        # connect_to_server_via_stdio(server_name: str, *, command: str, args: list[str], env: dict[str, str] | None = None, encoding: str = DEFAULT_ENCODING, encoding_error_handler: Literal["strict", "ignore", "replace"] = DEFAULT_ENCODING_ERROR_HANDLER) -> None: Connects to a server via stdio.
        # connect_to_server_via_sse(server_name: str, *, url: str) -> None: Connects to a server via SSE.
        # get_tools() -> list[BaseTool]: Retrieves all tools from connected servers.
    
    # langchain_mcp_adapters.tools.load_mcp_tools(session: ClientSession) -> list[BaseTool]: Loads MCP tools from a client session.
    
    # mcp.server.fastmcp.FastMCP: (MCP Server) A class for creating MCP servers in Python.
        # @mcp.tool(): Decorator to define tools.
        # run(transport: str): Starts the MCP server.
    
    # mcp.client.stdio.stdio_client: (MCP Client) Context manager for connecting to MCP servers via stdio.
    
    # mcp.client.sse.sse_client: (MCP Client) Context manager for connecting to MCP servers via SSE.
    

# Use Cases:
    # Extending LangChain Agents: Easily integrate tools from various MCP servers into LangChain agents.
    # Building Modular AI Systems: Create modular AI systems where different components (servers) provide specific functionalities (tools).
    # Integrating with Anthropic Models: Use MCP tools with Anthropic models (e.g., Claude) to enhance their capabilities.
    # Creating LangGraph Workflows with External Tools: Incorporate MCP tools into LangGraph workflows for complex AI applications.
# Example application

import asyncio
import nest_asyncio
import requests
from mcp.server.fastmcp import FastMCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import threading
import time
import io
import sys

nest_asyncio.apply()

# --- MCP Servers ---

def run_math_server():
    try:
        math_mcp = FastMCP("Math")

        @math_mcp.tool()
        def add(a: float, b: float) -> float:
            """Adds two numbers."""
            return a + b

        @math_mcp.tool()
        def multiply(a: float, b: float) -> float:
            """Multiplies two numbers."""
            return a * b

        math_mcp.run(transport="stdio")
    except Exception as e:
        print(f"Math server error: {e}")

def run_weather_server():
    try:
        weather_mcp = FastMCP("Weather")

        @weather_mcp.tool()
        def get_weather(city: str) -> str:
            """Gets the current weather for a city."""
            api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your API key
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            if data["cod"] == "404":
                return "City not found."
            else:
                description = data["weather"][0]["description"]
                temperature = data["main"]["temp"]
                return f"The weather in {city} is {description} with a temperature of {temperature}°C."

        weather_mcp.run(transport="stdio")
    except Exception as e:
        print(f"Weather server error: {e}")

# --- LangChain Agent ---

async def main():
    model = ChatOpenAI(model="gpt-4o")

    # Start servers in separate threads
    math_thread = threading.Thread(target=run_math_server)
    weather_thread = threading.Thread(target=run_weather_server)

    math_thread.daemon = True
    weather_thread.daemon = True

    math_thread.start()
    weather_thread.start()
    #Give the threads time to start.
    time.sleep(1)

    try:
      async with MultiServerMCPClient(
          {
              "math": {
                  "command": "python",
                  "args": ["-c", "pass"], #dummy command since server is already running.
                  "transport": "stdio",
              },
              "weather": {
                  "command": "python",
                  "args": ["-c", "pass"], #dummy command since server is already running.
                  "transport": "stdio",
              },
          }
      ) as client:
          tools = client.get_tools()
          agent = create_react_agent(model, tools)

          math_response = await agent.ainvoke({"messages": "What is 12 multiplied by 7?"})
          print("Math Response:", math_response)

          weather_response = await agent.ainvoke({"messages": "What is the weather in London?"})
          print("Weather Response:", weather_response)
    except Exception as e:
        print(f"Main error: {e}")

    #Threads will end when the main program ends.

if __name__ == "__main__":
    asyncio.run(main())
> Model Context Protocol (MCP)
# 1. What Is MCP and Why Use It?
    # Model Context Protocol (MCP) is a standard, open protocol that allows large language model (LLM) applications to securely integrate data, 
    # code, and external tools. You can think of MCP as a “universal interface” that both:
        # Servers implement to expose resources, prompts, and tools.
        # Clients use to discover and interact with these capabilities.
    # By adopting MCP, you can:
        # Integrate with many different backends or 3rd-party servers using the same protocol.
        # Expose your own custom logic (“tools”), data (“resources”), and prompts so that any LLM-based application—like Claude, ChatGPT, or a custom agent—can readily consume them.
        # Securely manage how and when LLMs see data or call external code.


# 2. Core MCP Concepts
    # MCP servers can provide three main categories of capabilities:
        # Resources – Data your server can provide. For example, database records, file contents, the output of an API call, or any other read-only data you want to pass to an LLM.
        # Tools – Functions or “actions” that an LLM can invoke with arguments. Tools can be used to write to a database, call an external API, or otherwise perform an operation.
        # Prompts – Reusable prompt templates and conversation setups that LLM-based applications can request.
    # Additionally, you’ll see advanced concepts like:
        # Images: Special resource or tool outputs that handle image data.
        # Sampling: Letting an MCP server itself call out to an LLM. (Often used in multi-LLM or “self-call” scenarios.)
        # Context: The idea that the LLM can load context from resources or prompt templates to handle a user request more effectively.


# 3. High-Level Workflow
    # When you connect an MCP client (like Claude for Desktop, or your own custom client) to an MCP server:
        #1. Initialization: The client and server exchange capabilities (e.g., “I support listing tools,” “I support subscribing to resource updates,” “I can do logging,” etc.).
        #2. Discovery: The client calls list_tools, list_resources, or list_prompts to find out what the server can do or provide.
        #3. Invocation:
            # If the LLM wants to read data, the client issues a read_resource.
            # If the LLM wants to do something (e.g., “create a new user”), the client calls tools/call.
            # If the LLM wants a prebuilt prompt template, the client calls prompts/get.
        #4. Response: The server returns results (resource data, tool outputs, prompt messages) back to the client, which then hands the content to the LLM in a structured way.


# ----------------------------------------------------------------------------------------------------------------------------

# 4. Building & Running a Server
    # Below is a minimal example of building your own Python-based MCP server using the mcp.server.fastmcp.FastMCP class. 
    # This approach is suitable when you want your code to run locally or be easily launched in a container/VM.

    # 4.1 Install MCP
        # Use uv or pip:
        uv add "mcp[cli]"
        # or
        pip install "mcp[cli]"

    # 4.1b Create a new environment
        conda create --name .mcp_server python=3.10
        conda activate .mcp_server
        # or
        python -m venv .venv
        .venv\Scripts\activate
        
    # 4.2 Example: Simple Calculator Server
        # file: calculator_server.py
        from mcp.server.fastmcp import FastMCP

        # Create an MCP server named "Calculator"
        mcp = FastMCP("Calculator")

        # # Tools can be normal Python functions with docstrings.
        @mcp.tool()
        def add(a: int, b: int) -> int:
            """
            Add two numbers
            """
            return a + b

        @mcp.tool()
        def multiply(a: int, b: int) -> int:
            """
            Multiply two numbers
            """
            return a * b

        if __name__ == "__main__":
            # By default, run with stdio transport (where MCP messages flow over stdin/stdout)
            mcp.run()   # default: transport="stdio" --> mcp.run(transport="stdio") or mcp.run(transport="sse")
        
    # How to run it in dev mode:
        # Launch an interactive inspector UI:
        mcp dev calculator_server.py
    
    # or test
        npx @modelcontextprotocol/inspector 

    # How to run it “directly”:
        # # Just run the server, so it reads/writes JSON over stdin/stdout
        mcp run calculator_server.py

    # How to integrate with Claude Desktop:
    mcp install calculator_server.py --name "My Calculator"
    # or edit the connection file directly: --> %APPDATA%\Claude\claude_desktop_config.json
    
    {
    "mcpServers": {
        "NBA_server": {
        "command": "C:\\Users\\pault\\MCP_SERVERS\\.venv\\Scripts\\python.exe", #or "C:\\Users\\pault\\MCP_SERVERS\\.venv\\bin\\python" if it exists,
        "args": [
            "C:\\Users\\pault\\MCP_SERVERS\\nba_server.py"
            ],
        "env": {
            "OPENAI_API_KEY": "<your-openai-api-key>"
            },
        "host": "127.0.0.1",
        "port": 8080,
        "timeout": 3000
            }
        }
    }

    # update these paths below with the above code for the client side
    """ 
    **Cursor** 
    `~/.cursor/mcp.json` 

    **Windsurf**
    `~/.codeium/windsurf/mcp_config.json`

    **Claude Desktop**
    `~/Library/Application\ Support/Claude/claude_desktop_config.json`

    **Claude Code**
    `~/.claude.json`
    
    """
    
    # This will update claude_desktop_config.json so Claude can see it as a new MCP server. Once it’s installed, open Claude Desktop, 
    # look for your server, and your two tools (add and multiply) will appear.


    # 4.3 Adding Resources
        # Let’s say you want to store some helpful text that LLMs might load. You can define a resource:
        @mcp.resource("readme://instructions")
        def instructions() -> str:
            """
            Basic usage instructions for the calculator
            """
            return "Use add(a,b) or multiply(a,b) to do arithmetic. Provide integer args!"
        # Now the LLM can request resources/read with the URI readme://instructions. Tools remain the same, but the LLM can load extra context 
        # from that resource.

    # 4.4 Adding a Prompt
        # Sometimes you want a prebuilt system prompt or conversation structure. You can define a prompt in code:
        @mcp.prompt()
        def troubleshoot_equation(equation: str) -> str:
            """Help debug a broken or invalid equation input."""
            return f"User is having trouble evaluating {equation}. Provide step-by-step help."
        # The client can discover this prompt by calling prompts/list, and then retrieve it with prompts/get.


# 5. Using a Third-Party MCP Server
    # You do not need to rewrite a remote server’s code if it already speaks MCP. You can simply spin up (or connect to) the third-party 
    # server, then point your client(s) to it. For example:

        # If you install a third-party “SQLite Explorer” server from GitHub, just run it:
        python sqlite_explorer_server.py
        # Then, from your client’s config or code, connect to that server over stdio or SSE.
        # Once connected, do a list_tools, see the tool named query_db, and call it with your SQL.

    # Integration example in code:
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp import ClientSession

        async def main():
            # Suppose the third-party server is a local python script
            # that we want to run with arguments...
            server_params = StdioServerParameters(
                command="python",
                args=["sqlite_explorer_server.py", "--db", "/path/to/data.db"]
            )

            # Establish a stdio-based connection:
            async with stdio_client(server_params) as (read_stream, write_stream):
                # Create a client session:
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize handshake
                    await session.initialize()

                    # List tools:
                    tools_response = await session.list_tools()
                    print("Tools:", tools_response.tools)

                    # For example, call the `query_db` tool
                    result = await session.call_tool("query_db", arguments={"sql": "SELECT * FROM foo"})
                    print("Query Results:", result)

        # Then run your async code
        import asyncio
        asyncio.run(main())

    # As soon as the third-party server is running, your client can discover and call all exposed Tools/Resources/Prompts. 
    # No special “hand-coded integration” is needed—just follow the MCP calls.


# 6. Writing an MCP Client
    # Most often, you’ll use an existing client, e.g. Claude for Desktop or your own agent framework, to talk to the server. 
    # If you want to code your own from scratch or from the Python SDK, here’s a quick snippet showing a typical usage pattern:

        # file: my_client.py
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def main():
            # 1. Describe how to launch the server
            server_params = StdioServerParameters(
                command="python",
                args=["calculator_server.py"],  # Our earlier example
                env={"SOME_ENV_VAR": "123"}     # if needed
            )

            # 2. Connect via stdio
            async with stdio_client(server_params) as (read_stream, write_stream):
                # 3. Create a high-level ClientSession
                async with ClientSession(read_stream, write_stream) as session:
                    # 4. Initialize the connection
                    await session.initialize()

                    # 5. List available tools
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        print("Tool discovered:", tool.name, tool.description)

                    # 6. Actually call a tool
                    sum_result = await session.call_tool("add", {"a": 4, "b": 5})
                    print("Sum result", sum_result)

                    # 7. Optionally read a resource
                    # read_result = await session.read_resource("readme://instructions")
                    # print("Resource contents:", read_result)

        # Run the client
        if __name__ == "__main__":
            asyncio.run(main())

    # What is happening behind the scenes?
        # We define how to run the server (in this case, “python calculator_server.py”).
        # We open a stdio connection via stdio_client(...).
        # We create ClientSession which handles the JSON-RPC handshake and method calls.
        # We do session.initialize(), which starts the “initialize” handshake with the server.
        # We discover tools/call and resources/read by listing them.
        # We call the add tool with some arguments. The server returns the sum as text.
        # We optionally do a resource read if needed.


# 7. Running the Entire System
    # Let’s suppose you want to see everything running end-to-end locally:

    # Terminal #1 – Launch your server:
        # For dev + interactive inspector
        mcp dev calculator_server.py

        # Or normal run in production mode
        mcp run calculator_server.py
        
    # Terminal #2 – Launch your custom client:
        python my_client.py

    # You’ll see logs from both sides. The client prints out discovered tools, calls them, and prints results. 
    # The server logs any invocation or resource read requests.


    # If you want to let Claude for Desktop manage the server, you’d do:
    mcp install calculator_server.py --name "My Calculator"
    # Then open Claude Desktop, confirm the new server appears, and try using the “tools panel” in Claude.


# 8. Examples & Advanced Usage
    # 8.1 Echo Server
    # A minimal server that echoes requests. Typically used for debugging. You can see the snippet below:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo")

        PATH = "C:/path/to/your/files/"
        
        @mcp.resource("docs://langgraph/full")
        def get_all_langgraph_docs() -> str:
            """
            Get all the LangGraph documentation. Returns the contents of the file llms_full.txt,
            which contains a curated set of LangGraph documentation (~300k tokens). This is useful
            for a comprehensive response to questions about LangGraph.

            Args: None

            Returns:
                str: The contents of the LangGraph documentation
            """

            # Local path to the LangGraph documentation
            doc_path = PATH + "llms_full.txt"
            try:
                with open(doc_path, 'r') as file:
                    return file.read()
            except Exception as e:
                return f"Error reading log file: {str(e)}"

        @mcp.tool()
        def echo_tool(msg: str) -> str:
            return f"Tool echo: {msg}"

        @mcp.prompt()
        def echo_prompt(message: str) -> str:
            return f"Please consider this message: {message}"

        if __name__ == "__main__":
            mcp.run()

    # 8.2 SQLite Explorer
        # See the python-sdk examples folder. It shows how to connect to a real DB, define resources for table schemas, and tools for queries.

    # 8.3 Using SSE Instead of Stdio
        # Sometimes you might prefer SSE (Server-Sent Events) over stdio. The MCP Python SDK includes an SSE transport, typically used within 
        # a Starlette or Uvicorn environment. You’d do:
        
        # inside a starlette route:
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from mcp.server.sse import SseServerTransport
        from mcp.server.lowlevel import Server

        server = Server("my-sse-server")
        sse_transport = SseServerTransport("/my_sse_endpoint/")

        async def sse_entrypoint(request):
            async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (rs, ws):
                await server.run(rs, ws, server.create_initialization_options())

        star_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=sse_entrypoint),
                Mount("/my_sse_endpoint/", app=sse_transport.handle_post_message)
            ]
        )
        # Then in your client code, you’d specify an SSE-based connection. This is more advanced usage.


# 9. Key Tips & Best Practices
    # Security: Tools can be extremely powerful. By default, anything the server implements is exposed to the LLM if your client permits it. 
        # Be mindful about restricting or gating certain calls or arguments.
    # Documentation: Provide docstrings and type hints in Python. The MCP Python SDK automatically extracts these to help LLMs understand 
        # your tools.
    # Testing: Use mcp dev server.py and the official MCP Inspector to check the list of tools, resources, and prompts. This interactive 
        # tool helps debug issues.
    # Environments: You can specify environment variables in mcp install server.py --env-file .env. That way, your secrets (API keys, 
        # DB credentials) remain separate from code.
    # Performance: For large or computationally heavy tasks, consider using progress notifications (ctx.report_progress(...)) so that the 
        # LLM or user gets partial updates.
    # Claude Desktop: Make sure you have the newest version. Use claude_desktop_config.json to define multiple servers. This is how you can 
        # run many local or remote servers at once, each focusing on different data or functionality.


# 10. Putting It All Together
    # 10.1 End-to-End Example
        # server.py (the server code we described, e.g. “Calculator”)
        # my_client.py (your custom client code, if you want it)
        # Claude for Desktop (another standard client)

    # Run the server:
    mcp run server.py

    # Install in Claude (optional but recommended if you use Claude Desktop):
    mcp install server.py --name "LocalCalculator"

    # Or run your custom client:
    python my_client.py

    # Interact:
        # If using the “Claude for Desktop” approach, you can open Claude, see “LocalCalculator,” and see a list of Tools. 
            # You might type “Please add 2 and 3,” and the LLM calls your add tool behind the scenes.
        # If using “my_client.py,” it will directly orchestrate calls to add, multiply, or resource reads.

    # You’ve now got a working MCP system where:
        # Your server defines custom logic and data (in Python).
        # The LLM client can discover and call that logic safely.
        # Tools or resources can easily be extended over time.
> Deploying Python MCP Server to Cloudflare with SSE
# refer to this video for more information: https://www.youtube.com/watch?v=cbeOWKANtj8
# refer to this second video: https://www.youtube.com/watch?v=H7Qe96fqg1M 
# refer to this documentation for more information: https://developers.cloudflare.com/agents/guides/remote-mcp-server
# refer to this github repository for more information: https://github.com/cloudflare/workers-mcp/tree/main
# reddit help: https://www.reddit.com/r/mcp/comments/1jjbgwu/hosting_mcp_on_the_cloud/ 

# Usage
    # Step 1: Generate a new Worker
        # Use "create-cloudflare? to generate a new Worker.
        """
        npx create-cloudflare@latest my-new-worker
        """
            # Use Hello World Worker
            # Use Typescript
            # Select "Yes" to use git for version control
            # Select "Yes" to deploy the application
            # When it starts to perform DNS propagation, use Ctrl+C to stop the process 
            

    # Step 2: Install workers-mcp
        """ 
        cd my-new-worker 
        npm install workers-mcp

        """
    # Step 3: Run the setup command (after deploying, this will also configure your claude desktop automatically)
        """ 
        npx workers-mcp setup
        """
            # Select "Yes" to deploy
            # Select "Yes" tp replace index.ts with the above code
            # You can change the name or use the same name for claude desktop
    
        # Error step:
            # If you get an error (■  ERROR spawn npx ENOENT), you can run the following command to fix it
                npx wrangler secret put SHARED_SECRET   #Then paste the value from your .dev.vars file when prompted.
            
            # You can also manually deploy it by running the following command:
                npx wrangler deploy
                

    # Step 4: You can then edit the index.ts file to include your server code
        # The code should include the server code as shown above
        
        # If you need to add environment variables, then edit the worker-configuration.d.ts and wrangler.toml files
            # worker-configuration.d.ts
                """
                interface Env {
                    // Tavily API key for search functionality
                    SHARED_SECRET: string;
                    TAVILY_API_KEY: string; # Add this line
                }
                """
            # wrangler.toml
                # option 1 (Using secrets (recommended for API keys)):
                    """
                    npx wrangler secret put TAVILY_API_KEY
                    """
                
                # option 2 (Using environment variables):
                    # Add the following to the wrangler.toml file
                    """
                    [vars]
                    TAVILY_API_KEY = "your-api-key"
                    
                    # You can then access the environment variable in your code as shown below
                    const tavily = new TavilyClient(env.TAVILY_API_KEY);
                    """
                    
        # Save the file and run the deploy command
        """
        npm run deploy
        """
    
    # Step 5: You can now interact with your server using Claude Desktop or your custom client
    npm install -g mcp-remote
    npm install mcp-remote

{
  "mcpServers": {
    "tavily-search": {
      "command": "C:\\Users\\pault\\Documents\\3. AI and Machine Learning\\2. Deep Learning\\1c. App\\Projects\\cloudflare_wrangler\\tavily\\tavily-search\\node_modules\\.bin\\mcp-remote",
      "args": [
        "https://tavily-search.tavily-search-mcp.workers.dev/sse"
      ]
    }
  }
}

{
  "mcpServers": {
    "tavily-search": {
      "command": "node",
      "args": [
        "C:\\Users\\pault\\Documents\\3. AI and Machine Learning\\2. Deep Learning\\1c. App\\Projects\\cloudflare_wrangler\\tavily\\tavily-search\\node_modules\\mcp-remote\\dist\\index.js",
        "https://tavily-search.tavily-search-mcp.workers.dev/sse"
      ]
    }
  }
}


    {
    "mcpServers": {
        "tavily-search": {
            "url": "https://tavily-search.tavily-search-mcp.workers.dev/sse"
            }
    }
    }
    

# # Deploying Python MCP Server to Cloudflare with SSE
# 
# ## 1. Using mcp-proxy to Expose Your Python Server via SSE
    # # This is the simplest approach to make your existing Python MCP server available over SSE

    # Install mcp-proxy
    uv tool install mcp-proxy
    # OR
    pipx install mcp-proxy

    # Run your existing Python MCP server behind the proxy with SSE support
    mcp-proxy --sse-port=8080 python sample.py

    # For public access (accessible outside localhost)
    mcp-proxy --sse-host=0.0.0.0 --sse-port=8080 python sample.py

    # Step 3: Configure your MCP client to connect
        # Your server will now be accessible at http://your-ip:8080/sse. If you're running this on a cloud VM or server 
        # with a public IP, you'll be able to connect to it from anywhere.

        {
        "mcpServers": {
            "my-remote-server": {
            "command": "mcp-proxy",
            "args": ["http://your-server-ip:8080/sse"],
            "env": {}
                }
            }
        }



# ## 2. Deploying to Cloudflare
    # # Based on https://developers.cloudflare.com/agents/guides/remote-mcp-server/
    # # Cloudflare doesn't directly support Python MCP servers yet, but offers proxy solutions

    # Install Cloudflare's Wrangler CLI
    npm install -g wrangler

    # Login to your Cloudflare account
    wrangler login

    # Create a new Cloudflare Worker project that will act as proxy
    npm create cloudflare@latest my-mcp-proxy -- --type="hello-world"
    cd my-mcp-proxy



# ## 3. Setting Up the Cloudflare Worker as a Proxy
    # # Create a Worker that proxies requests to your Python MCP server

    # Configure wrangler.toml - ensure you have these settings:
    # name = "my-mcp-proxy"
    # main = "src/index.js"
    # compatibility_date = "2024-03-22"
    # workers_dev = true

    # Deploy your Worker
    wrangler deploy

# ## 4. Configure MCP Clients to Connect to Your Remote Server
    # # Update your MCP client configuration to connect via mcp-proxy

    # For Claude Desktop (edit claude_desktop_config.json):
    # {
    #   "mcpServers": {
    #     "my-remote-server": {
    #       "command": "mcp-proxy",
    #       "args": ["https://my-mcp-proxy.your-account.workers.dev/sse"],
    #       "env": {}
    #     }
    #   }
    # }

# ## 5. Authentication for Remote MCP Server (Optional)
    # # From https://developers.cloudflare.com/agents/examples/build-mcp-server/

    # Create a new OAuth-enabled MCP server
    npm create cloudflare@latest -- my-mcp-server-github-auth --template=cloudflare/ai/demos/remote-mcp-github-oauth

    # Deploy the OAuth-enabled MCP server
    cd my-mcp-server-github-auth
    npm install
    npm run deploy

# ## 6. Docker Deployment Alternative
    # # If you prefer using Docker to host your Python MCP server with SSE

    # Create a Dockerfile for your Python MCP server with mcp-proxy
    # FROM python:3.10-slim
    # 
    # RUN pip install --no-cache-dir mcp-proxy
    # 
    # WORKDIR /app
    # COPY sample.py .
    # 
    # EXPOSE 8080
    # 
    # CMD ["mcp-proxy", "--sse-host=0.0.0.0", "--sse-port=8080", "python", "sample.py"]

    # Build and run the Docker container
    # docker build -t mcp-server .
    # docker run -p 8080:8080 mcp-server


# ## 7. Testing the Deployment
    # # Verify your SSE endpoint is working correctly

    # Use MCP Inspector to test a remote SSE endpoint
    npx @modelcontextprotocol/inspector
    # Then input your SSE URL (http://your-server-ip:8080/sse or https://your-worker.workers.dev/sse)



> MCP example (travel_server.py)
"""
Advanced MCP Server: Travel Planner

Features:
- Tools to search flights/hotels and build itineraries
- Resources that return JSON or text
- A reusable prompt template
- Lifecycle (lifespan) with DB initialization
- Uses progress notifications for a tool

Run:
    python travel_server.py
"""

import anyio
import httpx
import math
from typing import AsyncIterator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# MCP imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.utilities.logging import get_logger
from mcp.server.fastmcp.utilities.types import Image

# --------------------------------------------------------------------------
# LOGGING
logger = get_logger("travel_planner")

# --------------------------------------------------------------------------
# LIFESPAN CONTEXT (Pretend DB or external resources)
@dataclass
class TravelData:
    """In-memory store for flight/hotel info, could be replaced by real DB."""
    flight_db: dict = field(default_factory=dict)
    hotel_db: dict = field(default_factory=dict)

@asynccontextmanager
async def lifespan_ctx(server: FastMCP) -> AsyncIterator[TravelData]:
    """
    This lifespan is called once when the server starts and once when it stops.
    Use it to initialize DB connections, caches, etc.
    """
    logger.info("Travel Planner server is starting up...")
    # For demonstration, we just fill an in-memory dictionary.
    data = TravelData(
        flight_db={
            # flight_db is keyed by (origin, destination)
            ("SFO", "LAX"): [
                {"flight_no": "UA100", "price": 120, "duration": "1h25m"},
                {"flight_no": "DL223", "price": 150, "duration": "1h30m"}
            ],
            ("SFO", "NYC"): [
                {"flight_no": "UA300", "price": 300, "duration": "5h50m"},
                {"flight_no": "AA777", "price": 320, "duration": "5h45m"}
            ],
        },
        hotel_db={
            # hotel_db is keyed by city
            "Los Angeles": [
                {"name": "LA Grand Hotel", "stars": 5, "price_per_night": 250},
                {"name": "Sunset Motel", "stars": 3, "price_per_night": 90}
            ],
            "New York": [
                {"name": "NY Midtown Luxury", "stars": 5, "price_per_night": 350},
                {"name": "Queens Budget Inn", "stars": 2, "price_per_night": 70}
            ]
        }
    )
    try:
        yield data
    finally:
        logger.info("Travel Planner server is shutting down...")

# --------------------------------------------------------------------------
# CREATE MCP SERVER
mcp = FastMCP(
    name="Travel Planner",
    lifespan=lifespan_ctx,
    # Optionally declare dependencies that we want installed or recognized
    dependencies=["httpx", "anyio"]
)

# --------------------------------------------------------------------------
# RESOURCES
@mcp.resource("travel://top-destinations")
def get_top_destinations() -> str:
    """
    Returns a JSON string listing top travel destinations.
    For demonstration, it’s static. Typically you'd fetch from DB or API.
    """
    # Could also return a JSON string, or we could return python and let the server
    # convert, but here we'll just do direct JSON in a string
    return """
    {
      "destinations": [
        { "city": "Paris", "country": "France", "popularity": 9.9 },
        { "city": "Tokyo", "country": "Japan", "popularity": 9.8 },
        { "city": "New York", "country": "USA", "popularity": 9.6 }
      ]
    }
    """

@mcp.resource("travel://tips")
def get_general_travel_tips() -> str:
    """
    A plain text resource giving general travel tips.
    """
    return (
        "Always check the weather in your destination.\n"
        "Book flights & hotels in advance for better rates.\n"
        "Carry digital and physical copies of important documents.\n"
    )

# --------------------------------------------------------------------------
# TOOLS

@mcp.tool()
async def search_flights(origin: str, destination: str, ctx: Context) -> str:
    """
    Search flight info from origin to destination.
    Demonstrates usage of in-memory "flight_db" from lifespan context + progress.
    """
    data: TravelData = ctx.request_context.lifespan_context  # typed from our lifespan

    # We'll do a short loop with progress updates to simulate a longer process:
    for i in range(3):
        # "i" is progress index, 3 is total
        await ctx.report_progress(progress=i, total=3)
        await anyio.sleep(0.4)  # simulate network or DB query delay

    results = data.flight_db.get((origin.upper(), destination.upper()), [])
    if not results:
        return f"No flights found from {origin} to {destination}."
    msg = f"Flights from {origin} to {destination}:\n"
    for r in results:
        msg += (
            f"- Flight {r['flight_no']}, Price ${r['price']}, "
            f"Duration {r['duration']}\n"
        )
    return msg

@mcp.tool()
async def search_hotels(city: str, ctx: Context) -> str:
    """
    Search hotels in a given city. Also demonstrates usage of lifespan context.
    """
    data: TravelData = ctx.request_context.lifespan_context
    city_title = city.title()
    hotels = data.hotel_db.get(city_title, [])
    if not hotels:
        return f"No hotel data found for city: {city_title}"
    resp = f"Hotels in {city_title}:\n"
    for h in hotels:
        resp += (
            f"- {h['name']} | {h['stars']} star(s) | "
            f"${h['price_per_night']} per night\n"
        )
    return resp

@mcp.tool()
def build_itinerary(city: str, days: int, budget: float) -> str:
    """
    Build a simple itinerary for a given city, length of stay, and budget.
    This is a purely local function that doesn't call external APIs,
    but you'd typically combine flight/hotel costs or local events, etc.
    """
    # We'll do a naive approach: assume daily cost is 80% of budget/days
    daily_alloc = (budget / days) * 0.8
    rec = (
        f"Proposed itinerary for {days} days in {city.title()}:\n"
        f"- Daily budget (approx): ${daily_alloc:.2f}\n"
        f"- Activities: We'll schedule tours, local cuisine, etc.\n"
        f"- Suggestions: Book hotels earlier, keep some buffer for unexpected costs.\n"
    )
    return rec

# --------------------------------------------------------------------------
# PROMPTS
@mcp.prompt()
def plan_prompt(destination: str, additional_notes: str) -> str:
    """
    A prompt template that can be used by the LLM to refine a user's travel plan.

    Arguments:
        destination: The city or region the user wants to visit
        additional_notes: Additional user preferences or notes
    """
    return (
        "You are a travel-planning assistant. The user wants to visit "
        f"{destination.title()}. The user says:\n{additional_notes}\n\n"
        "Please propose a step-by-step plan or suggestions for them."
    )

# --------------------------------------------------------------------------
# SERVER RUN
if __name__ == "__main__":
    mcp.run()

> MCP example (travel_client.py)
"""
Example MCP Client for the Travel Planner server.

Connects via stdio to 'travel_server.py', initializes,
lists available tools/prompts/resources, and calls them.
"""

import asyncio

# From MCP Python SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 1) Define how to launch the server:
    server_params = StdioServerParameters(
        command="python",
        args=["travel_server.py"],
        env=None,  # Or pass in environment variables if needed
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        # 2) Create a high-level client session
        async with ClientSession(read_stream, write_stream) as session:
            # 3) Initialize handshake
            await session.initialize()

            print("\n--- MCP Client Initialized ---")

            # 4) List Tools
            tool_result = await session.list_tools()
            print("\nTools discovered:")
            for t in tool_result.tools:
                print(f" • {t.name}: {t.description}")

            # 5) Call a Tool, e.g. search_flights
            print("\n--- Calling search_flights Tool ---")
            # This tool is asynchronous; we can gather partial progress updates if we want
            # (the official sampling callback can handle progress notifications, but let's just do a single call here).
            flight_res = await session.call_tool("search_flights", arguments={
                "origin": "SFO",
                "destination": "LAX"
            })
            print("Flight search result:\n", flight_res)

            # 6) List Resources
            resources_result = await session.list_resources()
            print("\nResources discovered:")
            for r in resources_result.resources:
                print(f" • {r.uri} - {r.description}")

            # 7) Read Resource: travel://tips
            tip_content, tip_mime = await session.read_resource("travel://tips")
            print("\nTravel Tips Resource Content:\n", tip_content.decode("utf-8"))

            # 8) Use a prompt
            # Let's see what prompts exist
            prompts_resp = await session.list_prompts()
            print("\nPrompts discovered:")
            for p in prompts_resp.prompts:
                print(f" • {p.name}: {p.description}")

            # 9) Get the plan_prompt with arguments
            plan_prompt_resp = await session.get_prompt(
                name="plan_prompt",
                arguments={
                    "destination": "paris",
                    "additional_notes": "I want to focus on art, museums, and budget-friendly options."
                }
            )
            # The server responded with a set of messages
            print("\nPlan Prompt Messages:\n")
            for msg in plan_prompt_resp.messages:
                print(f"Role: {msg.role}, Content:\n{msg.content}")

            # 10) Optionally call 'build_itinerary' to see final
            itinerary_res = await session.call_tool("build_itinerary", arguments={
                "city": "Tokyo",
                "days": 5,
                "budget": 1200
            })
            print("\nSuggested itinerary:\n", itinerary_res)

            print("\n--- Done! ---")

if __name__ == "__main__":
    asyncio.run(main())

> MCP example 2
# Personalized Recipe Recommendation System using MCP
    # This server will:
        # Provide access to a recipe database (resource).
        # Offer tools for searching, filtering, and generating personalized recipes.
        # Include prompts for interacting with the recipe recommendation system.



# Project Structure:
recipe_server/
├── server.py
├── recipes.json
├── requirements.txt



# recipes.json # (Recipe Database):

[
  {
    "id": 1,
    "name": "Spaghetti Carbonara",
    "ingredients": ["spaghetti", "eggs", "guanciale", "pecorino romano", "black pepper"],
    "cuisine": "Italian",
    "dietary_restrictions": [],
    "instructions": "Cook spaghetti. Fry guanciale. Mix eggs and cheese. Combine everything. Serve with black pepper."
  },
  {
    "id": 2,
    "name": "Vegetarian Chili",
    "ingredients": ["kidney beans", "black beans", "tomatoes", "onions", "bell peppers", "chili powder"],
    "cuisine": "Mexican",
    "dietary_restrictions": ["vegetarian"],
    "instructions": "Sauté onions and peppers. Add beans and tomatoes. Season with chili powder. Simmer until thick."
  },
    {
    "id": 3,
    "name": "Chicken Stir-Fry",
    "ingredients": ["chicken breast", "broccoli", "carrots", "soy sauce", "ginger", "garlic"],
    "cuisine": "Asian",
    "dietary_restrictions": [],
    "instructions": "Cut chicken and vegetables. Stir-fry chicken. Add vegetables, soy sauce, ginger, and garlic. Cook until chicken is done."
  },
    {
    "id": 4,
    "name": "Vegan Chocolate Cake",
    "ingredients": ["flour", "sugar", "cocoa powder", "baking soda", "almond milk", "vegetable oil"],
    "cuisine": "Dessert",
    "dietary_restrictions": ["vegan"],
    "instructions": "Mix dry ingredients. Add almond milk and oil. Bake until done. Frost as desired."
  },
    {
    "id": 5,
    "name": "Salmon with Lemon Dill Sauce",
    "ingredients": ["salmon fillets", "lemon", "dill", "butter", "white wine"],
    "cuisine": "Seafood",
    "dietary_restrictions": [],
    "instructions": "Bake salmon fillets. Prepare lemon dill sauce with butter, lemon, dill, and white wine. Serve sauce over salmon."
  }
]



# server.py     # (MCP Server):
    import json
    from typing import List, Optional

    from mcp.server.fastmcp import FastMCP, Context
    from pydantic import BaseModel

    mcp = FastMCP("Recipe Recommendation Server")

    # Load recipes from JSON
    with open("recipes.json", "r") as f:
        recipes = json.load(f)

    class Recipe(BaseModel):
        id: int
        name: str
        ingredients: List[str]
        cuisine: str
        dietary_restrictions: List[str]
        instructions: str

    class RecipeFilter(BaseModel):
        cuisine: Optional[str] = None
        dietary_restrictions: Optional[str] = None
        ingredients: Optional[List[str]] = None

    @mcp.resource("recipes://all")
    def get_all_recipes() -> List[Recipe]:
        """Returns all recipes."""
        return [Recipe(**recipe) for recipe in recipes]

    @mcp.tool()
    def search_recipes(query: str, ctx: Context) -> List[Recipe]:
        """Searches recipes by name or ingredients."""
        query = query.lower()
        results = [
            Recipe(**recipe)
            for recipe in recipes
            if query in recipe["name"].lower() or any(query in ingredient.lower() for ingredient in recipe["ingredients"])
        ]
        ctx.info(f"Found {len(results)} recipes matching '{query}'")
        return results

    @mcp.tool()
    def filter_recipes(filters: RecipeFilter, ctx: Context) -> List[Recipe]:
        """Filters recipes by cuisine, dietary restrictions, or ingredients."""
        filtered_recipes = recipes[:]
        if filters.cuisine:
            filtered_recipes = [r for r in filtered_recipes if filters.cuisine.lower() == r["cuisine"].lower()]
        if filters.dietary_restrictions:
            filtered_recipes = [r for r in filtered_recipes if filters.dietary_restrictions.lower() in [d.lower() for d in r["dietary_restrictions"]]]
        if filters.ingredients:
            filtered_recipes = [r for r in filtered_recipes if all(ingredient.lower() in [i.lower() for i in r["ingredients"]] for ingredient in filters.ingredients)]

        ctx.info(f"Filtered to {len(filtered_recipes)} recipes.")
        return [Recipe(**recipe) for recipe in filtered_recipes]

    @mcp.tool()
    def generate_personalized_recipe(ingredients: List[str], ctx: Context) -> Recipe:
        """Generates a personalized recipe based on available ingredients."""
        # Simple logic: finds a recipe that uses the most provided ingredients.
        best_recipe = None
        max_matches = 0
        for recipe in recipes:
            matches = sum(1 for ingredient in ingredients if ingredient.lower() in [i.lower() for i in recipe["ingredients"]])
            if matches > max_matches:
                max_matches = matches
                best_recipe = recipe

        if best_recipe:
            ctx.info(f"Generated personalized recipe: {best_recipe['name']}")
            return Recipe(**best_recipe)
        else:
            ctx.info("No matching recipe found.")
            return None

    @mcp.prompt()
    def recipe_recommendation_prompt(user_preferences: str, ctx: Context) -> str:
        """Generates a prompt for recipe recommendations."""
        return f"Based on the user's preferences: '{user_preferences}', recommend a recipe."


# requirements.txt:
fastapi
uvicorn
pydantic
mcp


# To run the server:
    # Install dependencies: pip install -r requirements.txt
    # Run the server: mcp dev server.py


# MCP Client (Example using Python SDK):


from mcp.client.client import MCPClient
import asyncio

async def main():
    client = MCPClient("http://127.0.0.1:8000") # default dev server address
    await client.connect()

    # Get all recipes
    all_recipes = await client.get_resource("recipes://all")
    print("All Recipes:", all_recipes)

    # Search for recipes
    search_results = await client.call_tool("search_recipes", query="chicken")
    print("\nSearch Results:", search_results)

    # Filter recipes
    filter_results = await client.call_tool("filter_recipes", filters={"dietary_restrictions": "vegetarian"})
    print("\nFiltered Recipes:", filter_results)

    # Generate personalized recipe
    personalized_recipe = await client.call_tool("generate_personalized_recipe", ingredients=["tomatoes", "onions", "beans"])
    print("\nPersonalized Recipe:", personalized_recipe)

    # Use a prompt
    prompt_result = await client.call_prompt("recipe_recommendation_prompt", user_preferences="I want a quick and healthy dinner.")
    print("\nPrompt Result:", prompt_result)

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
> LangGraph Studio
# Step 1: Create the langgraph.json file (see example)
{
    "dockerfile_lines": [], 
    "graphs": {
        "chat": "./src/react_agent/graph.py:graph",
        "researcher": "./src/react_agent/graph.py:researcher",
        "agent": "./src/react_agent/graph.py:agent",
    },
    "env": [
        "OPENAI_API_KEY",
        "WEAVIATE_API_KEY",
        "WEAVIATE_URL",
        "ANTHROPIC_API_KEY",
        "ELASTIC_API_KEY"
    ],
    # or
    "env": "./.env",
    "python_version": "3.11",
    "dependencies": [
        "."
    ]
}

# Step 2: Run the langgraph-cli command
!pip install "langgraph-cli[inmem]==0.1.55" # Install the langgraph-cli package

# Step 3: Move to the directory containing the langgraph.json file

# Step 4: Install the dependencies
    # If you are using requirements.txt:
    python -m pip install -r requirements.txt

    # If you are using pyproject.toml or setuptools:
    python -m pip install -e .

# Step 5: Run the LangGraph server
langgraph dev # start a local development server
# or
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
        # uvx: python environment manager (like pyenv, conda, etc.). Creates isolated environments for Python applications.
        # --refresh: This flag tells uvx to refresh or recreate the environment
        # --from "langgraph-cli[inmem]": Specifies the source of the environment. In this case, it's from the langgraph-cli package.
        # --with-editable .: installs the current directory (your LangGraph application) in "editable" mode
        # --python 3.11: Specifies the Python version to use for the environment.
        # langgraph dev: This command starts the LangGraph server in n-memory mode.

> LangGraph Conditional Edges
#------------------------------------Basic Conditional Edge-------------------------------------
#-----------------------------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define the state structure
class State(TypedDict):
    value: int
    query: str
    response: str
    
# Define the conditional function
def conditional_edge(state: State) -> str:
    if state["value"] > 10:
        return "node_b"
    else:
        return "__end__"

# Define the graph
builder = StateGraph(State)
builder.add_node("node_a", lambda state: {"value": state["value"] + 1})
builder.add_node("node_b", lambda state: {"value": state["value"] - 1})
builder.add_edge(START, "node_a")
builder.add_conditional_edges("node_a", conditional_edge)
builder.add_edge("node_b", "node_a")
graph = builder.compile()

# Test the graph
initial_state = {"value": 5}
result = graph.invoke(initial_state)


#------------------------------------Router with Multiple Conditions-------------------------------------
#---------------------------------------------------------------------------------------------------------
members = ["researcher", "coder"]

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]

    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    return supervisor_node

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("search", search_node)
builder.add_node("web_scraper", web_scraper_node)
builder.add_edge(START, "supervisor")
graph = builder.compile()


#------------------------------------Using Tool Conditions-------------------------------------
#---------------------------------------------------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", ToolNode([retriever_tool]))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)



#------------------------------------Using Tool conditions from Scratch-------------------------------------
#-----------------------------------------------------------------------------------------------------------
def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"    # you can change this to any other node name instead of "__end__"


workflow.add_conditional_edges(
    "agent",
    tools_condition,  # decides if the agent is calling a tool or finishing
    {
        "tools": "retrieve",        # the dictionary is helpful if we named the nodes differently from the default tool condition function
        END: END,  # if the agent does not call any tool, we end the graph
    },
)

#------------------------------------Custom Condition functions-------------------------------------
#---------------------------------------------------------------------------------------------------

from typing import Literal, Union, List, Dict, Any
from langchain_core.messages import AnyMessage, HumanMessage

def data_api_condition(
    state: Union[List[AnyMessage], Dict[str, Any]],
    messages_key: str = "messages",
) -> Literal["data_api_node", "assistant_node"]:
    """
    Route to the Data API Node if the query involves fetching information.
    Otherwise, route to the Assistant Node.
    """
    if isinstance(state, list):
        user_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        user_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state: {state}")
    
    # Check if the query involves fetching information
    if isinstance(user_message, HumanMessage) and any(keyword in user_message.content.lower() for keyword in ["weather", "stock", "price"]):
        return "data_api_node"
    return "assistant_node"



#------------------------------------Custom Conditional Edges 2---------------------------------
#-----------------------------------------------------------------------------------------------

from langgraph.graph import END

def should_continue(state: AgentState) -> Literal["tools", "agent", END]:
    messages = state["messages"]
    if not messages:
        return "agent"  # Start the conversation
    
    last_message = messages[-1]
    
    # If the last message is from a tool
    if isinstance(last_message, ToolMessage):
        if last_message.content == "File added successfully":
            state["file_added"] = True
            print("📌 File addition confirmed")
            return "agent"
        print("🏁 Search complete, ending workflow")
        return END
    
    # If the last message is from the AI
    if isinstance(last_message, AIMessage):
        # If the file is added but not indexed, wait
        if state.get("file_added") and not state.get("indexed"):
            print("⏳ Waiting for indexing to complete...")
            return "agent"
        
        # If the AI asks to call a tool
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("🛠️ Executing tool calls...")
            return "tools"
    
    return "agent"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([tool_1, tool_2]))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",  # Route to tools if tool calls are detected
        "agent": "agent",  # Continue with the agent if no tool calls
        END: END,          # End the workflow if conditions are met
    }
)
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
> LangChain Messages (HumanMessage, AIMessage, SystemMessage, BaseMessage)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ChatMessage, ToolMessage, RemoveMessage

# BaseMessage
    # The base class for all message types. Inherited by HumanMessage, AIMessage, and SystemMessage
    class CustomMessage(BaseMessage):
        content: str
        role: str  # e.g., "user", "assistant", "system"

    custom_msg = CustomMessage(content="Hello, world!", role="user")

#------------------ Message Types -------------------------------------------------------------
#----------------------------------------------------------------------------------------------
# HumanMessage
    # Represents a message from a human user.
    human_msg = HumanMessage(content="My name is John Doe.", name = "Paul Okafor")  # you can use the name of the node or agent
    
    # or
    # Represents a message from a human user.
    class HumanMessage(BaseMessage):
        content: str

    human_msg = HumanMessage(content="My name is John Doe.")

# AIMessage
    # Represents a message generated by an AI agent.
    ai_msg = AIMessage(content="I am a helpful assistant.")

# SystemMessage
    # Represents a system message or prompt.
    system_msg = SystemMessage(content="You are a helpful assistant.")

# ChatMessage
    # Represents a message in a chat conversation.
    chat_msg = ChatMessage(role="custom_role", content="This is a custom message.")

# ToolMessage
    # Represents a message generated by a tool.
    tool_msg = ToolMessage(content="This is a tool message.", tool_call_id="123", tool_name="GradeMaster", id="123")

# RemoveMessage
    # Represents a message to remove a message from the conversation.
    remove_msg = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    
#---------------------------------- When to use it----------------------------------------------
#-----------------------------------------------------------------------------------------------
# Example 1:
messages = [SystemMessage(content="Welcome! Please provide your name.")]
user_input = "My name is John Doe."
messages.append(HumanMessage(content=user_input))
messages.append(AIMessage(content=responses))

# Example 2:
messages = [
    HumanMessage(content="My name is John Doe."),
    AIMessage(content="Hello, John Doe! How can I assist you today?"),
]

# Example 3: Prompt Template
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Welcome! Please provide your name."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chat_template.invoke({"messages": [HumanMessage(content="John Doe")]})
chat_template.messages

# Example 4: Agent Invocation
agent = create_react_agent(tools=tools, llm=llm)
messages = [    # Simulate a conversation
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Search for LangChain documentation."),
]

# Invoke the agent
response = agent.invoke({"messages": messages})


# Example 5: Node Example
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph

class OverallState(TypedDict):
    messages: Sequence[BaseMessage]

def user_node(state: OverallState) -> OverallState:
    try:
        # Initialize the conversation if no messages exist
        if not state.messages:
            state.messages = [SystemMessage(content="Welcome! Please provide your name.")]
        
        # Check if the last message is from the user (HumanMessage)
        if state.messages and isinstance(state.messages[-1], HumanMessage):
            # Invoke the LLM with the current state
            response = llm.invoke(state.messages)
            
            # Append the LLM's response as an Assistant Message (AIMessage)
            state.messages.append(AIMessage(content=response.content))
        
        return state
    except Exception as e:
        # Handle errors gracefully
        state.messages = state.messages + [SystemMessage(content=f"An error occurred: {str(e)}")]
        return state

> LangGraph Workflow
# BaseAgent : The base class for all agent nodes. Provides a common interface for all agents..
# LandGraphNode : A generic template for any node in the LandGraph.
# Workflow : Compiles and executes the workflow by connecting all nodes.


#--------------------------------------------- BaseAgent ---------------------------------------------
#-----------------------------------------------------------------------------------------------------

from typing import Dict, Any, List, Optional
from .utils.views import print_agent_output

class BaseAgent:
    """
    Base class for all agents. Provides common functionality and a standardized interface.
    """
    def __init__(self, websocket=None, stream_output=None, headers=None, tools: Optional[List[Any]] = None, **kwargs):
        """
        Initialize the agent.
        Args:
            websocket: WebSocket connection for real-time communication.
            stream_output: Function to stream output to the client.
            headers: Additional headers or metadata for the agent.
            tools: A list of tools (functions or objects) that the agent can use.
            **kwargs: Additional configuration for the agent.
        """
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tools = tools or []
        self.config = kwargs

    async def log(self, message: str, agent_name: str = "AGENT"):
        """
        Log messages to the console or stream them via WebSocket.
        Args:
            message: The message to log.
            agent_name: The name of the agent (for logging purposes).
        """
        if self.websocket and self.stream_output:
            await self.stream_output("logs", agent_name.lower(), message, self.websocket)
        else:
            print_agent_output(message, agent_name)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task. Subclasses must implement this method.
        Args:
            state: The current state of the workflow.
        Returns:
            Updated state after the agent's execution.
        """
        raise NotImplementedError("Subclasses must implement the `run` method.")

    def add_tool(self, tool: Any):
        """
        Add a tool to the agent's toolkit.
        Args:
            tool: A function or object that the agent can use.
        """
        self.tools.append(tool)

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Retrieve a tool by name or identifier.
        Args:
            tool_name: The name or identifier of the tool.
        Returns:
            The tool if found, otherwise None.
        """
        for tool in self.tools:
            if hasattr(tool, "__name__") and tool.__name__ == tool_name:
                return tool
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
        return None
    

#--------------------------------------------- LandGraphNode ---------------------------------------------
#---------------------------------------------------------------------------------------------------------

class LandGraphNode(BaseAgent):
    """
    A generic node template for the LandGraph. Can be customized for any task.
    """
    def __init__(self, node_name: str, **kwargs):
        """
        Initialize the node.
        Args:
            node_name: The name of the node (for identification and logging).
            **kwargs: Additional configuration for the node.
        """
        super().__init__(node_name=node_name,**kwargs)
        self.add_tool([tool_1, tool_2])

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and return the updated state.
        Args:
            state: The current state of the workflow.
        Returns:
            Updated state after processing.
        """
        query = state.get("task", {}).get("query", "")
        await self.log(f"Processing task in node: {query}", self.node_name.upper())

        # Use the tabular search tool
        tool = self.get_tool("tabular_search_tool")
        if tool:
            results = await tool(query, self.table)
            return {"node_name": self.node_name, "results": results}
        else:
            return {"node_name": self.node_name, "error": "Tool not found"}

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the node's task.
        Args:
            state: The current state of the workflow.
        Returns:
            Updated state after the node's execution.
        """
        return await self.process(state)





#--------------------------------------------- Workflow ----------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# This class is responsible for:
    # Initializing all nodes.
    # Defining the workflow graph.
    # Compiling and executing the workflow.

from langgraph.graph import StateGraph, END
import time

class Workflow:
    """
    The Workflow class compiles all nodes into a LandGraph and executes the workflow.
    """
    def __init__(self, task: Dict[str, Any], websocket=None, stream_output=None, headers=None):
        """
        Initialize the workflow.
        Args:
            task: The task to execute. Must include a "query" and can include additional metadata.
            websocket: WebSocket connection for real-time communication.
            stream_output: Function to stream output to the client.
            headers: Additional headers or metadata.
        """
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.task_id = self._generate_task_id()
        self.nodes = self._initialize_nodes()

    def _generate_task_id(self) -> int:
        """Generate a unique task ID."""
        return int(time.time())

    def _initialize_nodes(self) -> Dict[str, LandGraphNode]:
        """
        Initialize all nodes for the workflow.
        Returns:
            A dictionary of nodes, keyed by their names.
        """
        return {
            "node_1": LandGraphNode(node_name="node_1", websocket=self.websocket, stream_output=self.stream_output, headers=self.headers),
            "node_2": LandGraphNode(node_name="node_2", websocket=self.websocket, stream_output=self.stream_output, headers=self.headers),
            "node_3": LandGraphNode(node_name="node_3", websocket=self.websocket, stream_output=self.stream_output, headers=self.headers),
        }

    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the workflow graph using the initialized nodes.
        Returns:
            The compiled workflow graph.
        """
        workflow = StateGraph(ResearchState)

        # Add nodes to the graph
        for node_name, node in self.nodes.items():
            workflow.add_node(node_name, node.run)

        # Define edges between nodes
        workflow.add_edge("node_1", "node_2")
        workflow.add_edge("node_2", "node_3")
        workflow.set_entry_point("node_1")
        workflow.add_edge("node_3", END)

        return workflow

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow.
        Returns:
            The final result of the workflow.
        """
        workflow_graph = self._create_workflow_graph()
        compiled_workflow = workflow_graph.compile()

        await self.log(f"Starting workflow for task: {self.task.get('query')}", "WORKFLOW")
        result = await compiled_workflow.ainvoke({"task": self.task})
        return result


#----------------------------------------------
# call the workflow
#----------------------------------------------
task = {
    "query": "Process some data",  # The main query or input
    "verbose": True,               # Optional: Whether to log detailed output
}

# Initialize the workflow
workflow = Workflow(task=task)

# Execute the workflow
result = await workflow.execute()
print(result)
> LangSmith

# LangSmith is a platform by LangChain that helps developers trace, debug, and evaluate LLM (Large Language Model) applications. 
# It provides tools for observability (seeing how your app works), testing (ensuring your app behaves as expected), and feedback collection 
# (improving your app based on user input).

# Why is LangSmith Useful?
    # Tracing : See how your app processes inputs and generates outputs (Logs every step of your app's execution).
    # Testing : Evaluate your app's performance with datasets (Runs your app on datasets to measure performance).
    # Feedback : Collect user feedback to improve your app.
    # Observability : Monitor your app in production/real-time to catch issues early.
# --------------------------------- Setting Up LangSmith ---------------------------------
import os
from langsmith import Client, traceable, wrappers
from langchain.smith import RunEvalConfig, run_on_dataset


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_PROJECT"] = "my-project" 
# Load environment variables from a .env file (optional)
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env", override=True)



# --------------------------------- Tracing with LangSmith ---------------------------------
# @traceable Decorator : Automatically logs function calls.
# Use trace context manager for specific blocks of code.

from langsmith import traceable

# Use @traceable to log function calls
@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

# Use context manager for fine-grained tracing
from langsmith import trace
with trace(name="Generate Response", run_type="chain") as ls_trace:
    response = call_openai(messages)
    ls_trace.end(outputs={"output": response})

# Wrap OpenAI client for automatic tracing
from langsmith.wrappers import wrap_openai
openai_client = wrap_openai(openai.Client())



# --------------------------------- Testing and Evaluation with LangSmith ---------------------------------
# Use create_dataset to create and manage datasets.
# Define custom evaluators to score your app's outputs.
# Use evaluate to run experiments and measure performance.

from langsmith import Client

# Create a dataset
client = Client()

#---------------------------------
# Create a dataset
#---------------------------------

# Create dataset for testing our AI agents
dataset_input = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "Who wrote the book '1984'?",  "output": "George Orwell"},
    {"input": "What is the square root of 16?",  "output": "4"},
]

dataset = client.create_dataset(
    dataset_name = "my-dataset", 
    description="A dataset for testing AI agents.")

for data in dataset_input:
    try:
        client.create_example(
            inputs={"question": data['input']},  # Wrapping the input into a dictionary
            outputs={"answer": data['output']},  # Wrapping the output into a dictionary
            dataset_id=dataset.id  # Assuming dataset.id is already created
        )
    except Exception as e:
        print(f"Failed to create example for input: {data['input']}, Error: {e}")
    
# or use create_examples to add multiple examples at once
client.create_examples(
    inputs=[{"question": data['input']}],
    outputs=[{"output": data['output']}],
    dataset_id=dataset.id
)

#---------------------------------
# Create a Target or label
#---------------------------------
# Define the application logic you want to evaluate inside a target function
# The SDK will automatically send the inputs from the dataset to your target function
def target(inputs: dict) -> dict:
  response = openai_client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          { "role": "system", "content": "Answer the following question accurately" },
          { "role": "user", "content": inputs["question"] },
      ],
  )
  return { "response": response.choices[0].message.content.strip() }


#---------------------------------
# Define an Evaluator
#---------------------------------

# Define instructions for the LLM judge evaluator
instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
- False: No conceptual match and similarity
- True: Most or full conceptual match and similarity
- Key criteria: Concept should match, not exact wording.
"""

# Define output schema for the LLM judge
class Grade(BaseModel):
  score: bool = Field(
      description="Boolean that indicates whether the response is accurate relative to the reference answer"
  )

# Define LLM judge that grades the accuracy of the response relative to reference output
def accuracy(outputs: dict, reference_outputs: dict) -> bool:
  response = openai_client.beta.chat.completions.parse(
      model="gpt-4o-mini",
      messages=[
          { "role": "system", "content": instructions },
          {
              "role": "user",
              "content": f"""Ground Truth answer: {reference_outputs["answer"]}; 
              Student's Answer: {outputs["response"]}"""
          },
      ],
      response_format=Grade,
  )
  return response.choices[0].message.parsed.score



#---------------------------------
# Run and View results
#---------------------------------
# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
  target,
  data="my-dataset",
  evaluators=[
      accuracy,
      # can add multiple evaluators here
  ],
  experiment_prefix="first-eval-in-langsmith",
  max_concurrency=2,
)





# --------------------------------- Prompt Engineering ---------------------------------

from langsmith import Client
from langsmith.client import convert_prompt_to_openai_format

# Pull a prompt from LangSmith Prompt Hub
client = Client()
prompt = client.pull_prompt("your-prompt-id")

# Use the prompt in your app
hydrated_prompt = prompt.invoke({"question": "What is LangSmith?"})
messages = convert_prompt_to_openai_format(hydrated_prompt)["messages"]
response = openai_client.chat.completions.create(model="gpt-4", messages=messages)




# --------------------------------- Collecting Human Feedback ---------------------------------
# Add feedback to a run
from langsmith import Client
from langsmith import traceable
import uuid

@traceable
def foo():
    return "This is a sample Run!"


client = Client()
client.create_feedback(
    run_id="your-run-id",
    key="user_feedback",
    score=1.0,
    comment="The response was helpful."
)

# Pre-generate run IDs for feedback
pre_defined_run_id = uuid.uuid4()
foo(langsmith_extra={"run_id": pre_defined_run_id})
client.create_feedback(pre_defined_run_id, "user_feedback", score=1)



# --------------------------------- Production Observability ---------------------------------
# Filter runs in production
from langsmith import Client
from datetime import datetime, timedelta

client = Client()
runs = client.list_runs(
    project_name="langsmith-academy",
    filter="eq(is_root, true)",
    start_time=datetime.now() - timedelta(days=1)
)

for run in runs:
    print(run)

# Run your app to trigger online evaluations
from app import langsmith_rag
question = "How do I set up tracing?"
langsmith_rag(question)

# Evaluating LLM Project using LangSmith

# Step 1:
# Install dependencies
%%capture --no-stderr
%pip install langsmith langchain-openai langchain-core langchain-community pydantic python-dotenv openai
%pip install --upgrade langsmith


# Step 2:
# Perform the extraction (this here is what you want the LLM project to do)
    import langsmith
    from pydantic import BaseModel, Field
    from langsmith import wrappers, Client
    from openai import OpenAI
    openai_client = wrappers.wrap_openai(OpenAI())

    class UsefulInformation(BaseModel):
        products_and_services: list[str] = Field(description="A list of products and services provided by the company")
        risk_factors: list[str] = Field(description="A list of risk factors described in the document")
        irs_employer_id_number: list[str] = Field(description="The IRS Employer Identification Number of the company")
        company_address: list[str] = Field(description="The address of the company")
        earnings_per_share_basic: list[str] = Field(description="The basic earnings per share of the company")
        net_income: list[str] = Field(description="The net income of the company")

    def extract_information(doc):
        prompt = f"""
        The text below is an excerpt from a 10-K report. You must extract specific information and return it in a structured format.
        
        CRITICAL INSTRUCTIONS:
        1. AVOID DUPLICATES: Never include duplicate items in any list
        2. BE CONCISE: Keep each item brief and to the point
        3. VALIDATE: Each piece of information must be explicitly stated in the text, do not make assumptions
        4. FORMAT: All fields must be lists, even if empty or single item
        
        Examples of GOOD responses:
        - Products: ["Google Search", "Google Cloud", "Android"]
        - Address: ["1600 Amphitheatre Parkway, Mountain View, CA 94043"]
        - Phone: ["+1 650-253-0000"]
        
        Examples of BAD responses (avoid these):
        - Duplicates: ["Google Search", "Search by Google", "Google Search Engine"]
        - Too verbose: ["Google Search is a web search engine that allows users to search the World Wide Web..."]
        - Made up data: Do not include information unless explicitly found in the text
        
        Please extract:
        1. Products and Services: List unique products/services (max 10 items)
        2. Risk Factors: List unique, critical risks (max 10 items)
        3. IRS Employer ID Number: List any EIN found
        4. Company Address: List primary address of the company
        5. Earnings Per Share (Basic): List basic EPS figure
        6. Net Income: List net income figure

        Text from the 10-K report:
        {doc}
        """
        try:
            response = openai_client.beta.chat.completions.parse(
            model="o1-2024-12-17",
            messages=[
                { "role": "user", "content": prompt },
            ],
            response_format=UsefulInformation
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in structured output LLM call: {str(e)}")
            print(f"Error type: {type(e)}")
            return UsefulInformation(
                products_and_services=[],
                risk_factors=[],
                irs_employer_id_number=[],
                company_address=[],
                earnings_per_share_basic=[],
                net_income=[]
            )

    def process_all_docs():
        all_text =  load_pdf()
        results =  extract_information(all_text)
        print("processed all docs...")
        return results

    aggregated_info = process_all_docs()
    print(aggregated_info)


# Step 3:
# Define application logic to be evaluated
    from langsmith import traceable

    client = Client()

    @traceable
    def target(inputs: dict) -> dict:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                { "role": "user", "content": inputs["input"][0]["content"] },
            ],
            response_format=UsefulInformation
        )
        return { "response": response.choices[0].message.content }


# Step 4:
# Define evaluator
    import json

    def format_objects_for_llm_judge(obj1, obj2):
        """Formats two objects into natural language for easier LLM comparison."""
        def format_single_object(obj, object_name):
            if isinstance(obj, str):
                obj = json.loads(obj)
            formatted_sections = []
            formatted_sections.append(f"\n{object_name} contains the following information:")
            sorted_keys = sorted(obj.keys())
            for key in sorted_keys:
                values = obj[key]
                readable_key = key.replace('_', ' ').capitalize()
                if isinstance(values, list):
                    if len(values) == 1:
                        formatted_sections.append(f"\n{readable_key}: {values[0]}")
                    else:
                        items = '\n  - '.join(values)
                        formatted_sections.append(f"\n{readable_key}:\n  - {items}")
                else:
                    formatted_sections.append(f"\n{readable_key}: {values}")
            
            return '\n'.join(formatted_sections)

        object1_text = format_single_object(obj1, "Actual Output")
        object2_text = format_single_object(obj2, "Reference Output")
        return [object1_text, object2_text]

    @traceable(run_type="llm")
    def run_llm_judge(formatted_text):
        class Score(BaseModel):
            """Evaluate how well an extracted output matches a reference ground truth for 10-K document information."""
            accuracy: bool = Field(
                description=(
                    "A binary score (0 or 1) that indicates whether the model's extraction adequately matches the reference ground truth. "
                    "Score 1 if the model's output captures the same essential business information as the reference extraction, even if "
                    "expressed differently. The goal is to verify that the model successfully extracted similar key business information "
                    "as found in the reference ground truth, not to ensure identical representation."
                )
            )
            reason: str = Field(
                description=(
                    "An explanation of how well the model's extraction aligns with the reference ground truth. Consider how effectively "
                    "the model captured the same key business information, financial data, and risk factors as the reference output. "
                    "Acknowledge that variations in expression are acceptable as long as the same core information is captured."
                )
            )
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are evaluating how well a model's extraction of 10-K document information matches a reference ground truth output. "
                        "Your task is to determine if the model successfully captured the same essential business information as the reference, "
                        "understanding that similar concepts may be expressed differently.\n\n"
                        "Context:\n"
                        "- The reference output represents the ground truth extraction from a 10-K document\n"
                        "- The model's output is being evaluated against this reference for accuracy and completeness\n"
                        "- Both extractions contain key business information like products/services, risk factors, and financial metrics\n"
                        "- The goal is to verify the model captured similar information as the reference, not identical expression\n\n"
                        "Evaluation Guidelines:\n"
                        "- Score 1 (true) if the model's output:\n"
                        "  * Captures the same core business information as the reference\n"
                        "  * Identifies similar risk factors, even if described differently\n"
                        "  * Extracts matching or equivalent financial metrics\n"
                        "  * Contains consistent company identifiers\n"
                        "  * May include additional valid information beyond the reference\n\n"
                        "- Score 0 (false) only if the model's output:\n"
                        "  * Misses or contradicts critical information from the reference\n"
                        "  * Shows fundamental misunderstanding of the business details\n"
                        "  * Contains irreconcilable differences in key metrics\n"
                        "  * Fails to capture the essential information found in the reference\n\n"
                        "Remember: The reference output is our ground truth. Evaluate how well the model's extraction "
                        "captures the same essential business information, allowing for variations in expression.\n\n"
                        "Outputs to Evaluate:\n"
                        f"- **Model Output:** {formatted_text[0]}\n"
                        f"- **Reference Ground Truth:** {formatted_text[1]}\n"
                    )
                }
            ],
            response_format=Score
        )
        response_object = json.loads(response.choices[0].message.content)
        return { "response": response_object }

    @traceable
    def evaluate_accuracy(outputs: dict, reference_outputs: dict) -> dict:
        actual_output = outputs["response"]
        expected_output = reference_outputs['output']
        formatted_text = format_objects_for_llm_judge(actual_output, expected_output)
        object_response = run_llm_judge(formatted_text)["response"]
        return { "key": "accuracy",
                "score": object_response["accuracy"],
                "reason": object_response["reason"] }


# write code to calculate other LLM evaluation metrics
    # e.g. Tool Selection Accuracy, Task Completion Rate, End-to-End Latency, Token Efficiency etc.




# Step 5:
# Run evaluation
    experiment_results = client.evaluate(
        target,
        data="10-k extraction",
        evaluators=[evaluate_accuracy], # Add the evaluator function here
        experiment_prefix="10-k-extraction-gpt-4o", # The prefix for the experiment name
        max_concurrency=5,  # The number of concurrent evaluations to run
        num_repetitions=3   # The number of times to repeat the evaluation for each input
    )

    experiment_results.to_pandas()



> AgentOps
# Adding AgentOps to LangChain applications

# Install the AgentOps SDK and the additional LangChain dependency
pip install agentops
pip install agentops[langchain]

# Set up your import statements
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agentops.langchain_callback_handler import LangchainCallbackHandler



# Set up your LangChain handler to make the calls
handler = LangchainCallbackHandler(api_key=AGENTOPS_API_KEY, tags=['LangChain Example'])

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
	callbacks=[handler],
	model='gpt-3.5-turbo')

agent = initialize_agent(tools,
	llm,
	agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True,
	callbacks=[handler], # You must pass in a callback handler to record your agent
	handle_parsing_errors=True)
> Human in the Loop
# Use the `interrupt` function instead.

#--------------------------- Human Feedback Node -------------------------------------------------------
def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
    
    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
    
#------------------------- Basic Human-in-the-Loop with Breakpoints--------------------------------
# Compile graph with breakpoint
graph = builder.compile(
    checkpointer=memory, 
    interrupt_before=["step_for_human_in_the_loop"] # Add breakpoint
)

# Run graph up to breakpoint
thread_config = {"configurable": {"thread_id": "1"}}
for event in graph.stream(inputs, thread_config, stream_mode="values"):
    print(event)

# Perform human action (e.g., approve, edit, input)
# Resume graph execution
# Human approval step
user_approval = input("Do you want to call the tool? (yes/no): ")
if user_approval.lower() == "yes":
    # Resume graph execution
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
else:
    print("Operation cancelled by user.")

#------------------------- Dynamic Breakpoints--------------------------------
# Dynamic breakpoints allow the graph to interrupt itself based on conditions defined within a node.
# can define some *condition* that must be met for a breakpoint to be triggered
from langgraph.errors import NodeInterrupt

# Define a node with dynamic breakpoint
def my_node(state: State) -> State:
    if len(state['input']) > 5:  # Condition for breakpoint
        raise NodeInterrupt(f"Input too long: {state['input']}")
    return state

# Resume after dynamic breakpoint
graph.update_state(config=thread_config, values={"input": "foo"})  # Update state to pass condition
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)

# Skip node entirely
graph.update_state(None, config=thread_config, as_node="my_node")  # Skip node
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
    

#------------------------- Editing State with Human Feedback--------------------------------
# You can modify the graph state during interruptions to incorporate human feedback.

# Get current state after interruption
state = graph.get_state(thread_config)
print(state)

# Update state with human feedback
graph.update_state(
    thread_config, 
    {"user_input": "human feedback"},  # Add human input
    as_node="human_input"  # Treat update as a node
)

# Resume graph execution
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)


#------------------------- Input Pattern or Tool Call--------------------------------
    
 # Compile graph with input breakpoint
graph = builder.compile(
    checkpointer=checkpointer, 
    interrupt_before=["human_input"]  # Node for human input
)

# Run graph up to input breakpoint
for event in graph.stream(inputs, thread_config, stream_mode="values"):
    print(event)

# Add human input and resume
graph.update_state(
    thread_config, 
    {"user_input": "human input"},  # Provide human input or tool call
    as_node="human_input"  # Treat update as node
)
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)   
    


#------------------------- LangGraph API Integration--------------------------------
from langgraph_sdk import get_client

# Connect to LangGraph Studio
client = get_client(url="http://localhost:56091")

# Stream graph with breakpoint
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=initial_input,
    stream_mode="values",
    interrupt_before=["tools"],  # Set breakpoint
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)

# Resume from breakpoint
async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="values",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)


> LangGraph Memory Module
#------------------------- Short term Memory --------------------------------
# Short-term memory is managed using checkpointers , which save the state of a graph at each step.

from langgraph.checkpoint.memory import MemorySaver

# Initialize a checkpointer for short-term memory
checkpointer = MemorySaver()

# Compile a graph with the checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Run the graph and save state
thread_config = {"configurable": {"thread_id": "1"}}
for event in graph.stream(inputs, thread_config, stream_mode="values"):
    print(event)

# Retrieve the state from the checkpointer
state = graph.get_state(thread_config)
print(state)


#------------------------- Long-Term Memory with Stores (Memory Store) --------------------------------
# Long-term memory is managed using stores , which persist data across threads or sessions.
from langgraph.store.memory import InMemoryStore

# Initialize a store for long-term memory
store = InMemoryStore()

#---------------------
# Save a memory
#----------------------
# Save a memory
user_id = "1"
namespace = (user_id, "memories") # Namespace for user-specific memories
key = "profile"
value = {"name": "Lance", "interests": ["biking", "bakeries"]}
store.put(namespace, key, value)    # Save a memory to the store
#---------------------
#---------------------


#---------------------
# Save a memory 2 (Memory Schema Collection)
#---------------------
from pydantic import BaseModel, Field

# Define a memory schema
class Memory(BaseModel):
    content: str = Field(description="The main content of the memory.")

# Create a collection of memories
memory_collection = [
    Memory(content="User likes biking"),
    Memory(content="User enjoys bakeries")
]

# Save memories to the store
for memory in memory_collection:
    key = str(uuid.uuid4())
    store.put(namespace, key, memory.model_dump())
#---------------------
#---------------------

#---------------------------------
# Retrieve a memory from the store
#---------------------------------
memories = store.get(namespace, key)
print(memories.value)
# or
memories = store.search(namespace)
for memory in memories:
    print(memory.value)

#---------------------
#---------------------

#---------------------------------
# Dynamic Memory Updates
#---------------------------------
# Dynamic memory updates allow the agent to decide when to save memories and what type of memory to update 
# (e.g., profile, collection, or instructions)

def update_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    existing_memories = store.search(namespace)
    tool_name = "Memory"
    existing_memories_formatted = [(m.key, tool_name, m.value) for m in existing_memories]
    result = trustcall_extractor.invoke({"messages": state["messages"], "existing": existing_memories_formatted})
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, rmeta.get("json_doc_id", str(uuid.uuid4())), r.model_dump())



#---------------------------------------------- Memory Agents ------------------------------------------
#-------------------------------------------------------------------------------------------------------
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Define nodes for the agent
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Retrieve memory and personalize responses
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    memories = store.search(namespace)
    memory_content = "\n".join([m.value["content"] for m in memories])
    system_msg = f"Memory: {memory_content}"
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    # Reflect on chat history and save memories
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    result = trustcall_extractor.invoke({"messages": state["messages"]})
    for r in result["responses"]:
        key = str(uuid.uuid4())
        store.put(namespace, key, r.model_dump())

# Compile the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Compile with memory store and checkpointer
graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())





#------------------------- Long-Term Memory with Stores (Memory Store) --------------------------------









#------------------------- Long-Term Memory with Stores (Memory Store) --------------------------------




> LangMem (for Memory Storage)
# LangMem helps agents learn and adapt from their interactions over time.

# install the LangMem package
pip install -U langmem


# Creating an Agent
    # Import core components (1)
    from langgraph.prebuilt import create_react_agent
    from langgraph.store.memory import InMemoryStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

    # Set up storage (2)
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    ) 

    # Create an agent with memory capabilities (3)
    agent = create_react_agent(
        "anthropic:claude-3-5-sonnet-latest",
        tools=[
            # Memory tools use LangGraph's BaseStore for persistence (4)
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
        ],
        store=store,
    )


# The memory tools work in any LangGraph app. Here we use create_react_agent to run an LLM with tools, but you can add these tools to 
# your existing agents or build custom memory systems without agents.

# InMemoryStore keeps memories in process memory—they'll be lost on restart. For production, use the AsyncPostgresStore or a similar 
# DB-backed store to persist memories across server restarts.

# The memory tools (create_manage_memory_tool and create_search_memory_tool) let you control what gets stored. The agent extracts 
# key information from conversations, maintains memory consistency, and knows when to search past interactions.




# Using AsyncPostgresStore

    #1. Basic setup and usage:
    from langgraph.store.postgres import AsyncPostgresStore

    conn_string = "postgresql://user:pass@localhost:5432/dbname"

    async with AsyncPostgresStore.from_conn_string(conn_string) as store:
        await store.setup()  # Run migrations. Done once

        # Store and retrieve data
        await store.aput(("users", "123"), "prefs", {"theme": "dark"})
        item = await store.aget(("users", "123"), "prefs")
        
        
        
    #2. Vector search using LangChain embeddings:
    from langchain.embeddings import init_embeddings
    from langgraph.store.postgres import AsyncPostgresStore

    conn_string = "postgresql://user:pass@localhost:5432/dbname"

    async with AsyncPostgresStore.from_conn_string(
        conn_string,
        index={
            "dims": 1536,
            "embed": init_embeddings("openai:text-embedding-3-small"),
            "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
        }
    ) as store:
        await store.setup()  # Run migrations. Done once

        # Store documents
        await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
        await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})
        await store.aput(("docs",), "doc3", {"text": "Other guide"}, index=False)  # don't index

        # Search by similarity
        results = await store.asearch(("docs",), query="programming guides", limit=2)




    #3. Using connection pooling for better performance:
    from langgraph.store.postgres import AsyncPostgresStore, PoolConfig

    conn_string = "postgresql://user:pass@localhost:5432/dbname"

    async with AsyncPostgresStore.from_conn_string(
        conn_string,
        pool_config=PoolConfig(
            min_size=5,
            max_size=20
        )
    ) as store:
        await store.setup()  # Run migrations. Done once
        # Use store with connection pooling...


> LangGraph Project 1 (code that works)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import json # Parse JSON response



class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            streaming=True,
            callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
    def call_tool(self):
        tool = TavilySearchResults(max_results=2)
        self.tools = [tool]
        self.tool_node = ToolNode(tools=[tool])
        self.llm_with_tool = self.llm.bind_tools(self.tools)
        
    def call_model(self, state: MessagesState):
        """
        LLM node to process the user's query and invoke tools if needed.
        """
        messages = state['messages']
        latest_query = messages[-1].content if messages else "No query provided."

        # template = ''' 
        # You are a travel suggestion agent. Answer the user's questions based on their travel preferences. 
        # If you need to find information about a specific destination, use the search_tool. Understand that the information was retrieved from the web,
        # interpret it, and generate a response accordingly.

        # Answer the following questions as best as you can. You have access to the following tools:
        # {tools}

        # Use the following format:
        # Question: the input question you must answer
        # Thought: you should always think about what to do
        # Action: the action you should take, should be one of [{tool_names}]
        # Action Input: the input to the action
        # Observation: the result of the action
        # Thought: I now know the final answer
        # Final Answer: [Your final answer here as a concise and complete sentence]

        # Ensure the response strictly follows this format. Do not repeat the Final Answer multiple times.

        # Begin!
        # Question: {input}
        # Thought: {agent_scratchpad}
        # '''

        # Improved prompt to enforce JSON output
        template = ''' 
        You are a travel suggestion agent. Answer the user's questions based on their travel preferences. 
        If you need to find information about a specific destination, use the search_tool. Understand that the information was retrieved from the web,
        interpret it, and generate a response accordingly.

        Answer the following questions as best as you can. You have access to the following tools:
        {tools}

        Use the following format:
        
        "Question": the input question you must answer
        "Thought": your reasoning about what to do next
        "Action": the action you should take, one of [{tool_names}] (if no action is needed, write "None")
        "Action Input": the input to the action (if no action is needed, write "None")
        "Observation": the result of the action (if no action is needed, write "None")
        "Thought": your reasoning after observing the action
        "Final Answer": the final answer to the original input question
        
        Ensure every Thought is followed by an Action, Action Input, and Observation. If no tool is needed, explicitly write "None" for Action, Action Input, and Observation.

        Begin!
        Question: {input}
        Thought: {agent_scratchpad}
        '''
        
        prompt = PromptTemplate.from_template(template)
        search_agent = create_react_agent(
            llm=self.llm_with_tool,
            prompt=prompt,
            tools=self.tools
        )

        agent_executor = AgentExecutor(
            agent=search_agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=False,
            handle_parsing_errors=True,
        )

        try:
            response = agent_executor.invoke({
                "input": latest_query,
                "agent_scratchpad": ""  # Initialize with an empty scratchpad
            })

            # Check if the response is already a dictionary
            if isinstance(response, dict):
                final_answer = response.get("output", "No final answer provided.")
            else:
                raise ValueError("Unexpected response type. Expected a dictionary.")

            # print("")
            # print(f'response: {response}')
                
            # # Validate and clean response
            # if not response.startswith("Final Answer:"):
            #     raise ValueError("Invalid agent response format. Missing 'Final Answer:' prefix.")
            # final_answer = response.replace("Final Answer:", "")[-1].strip()
            
            state['messages'].append(AIMessage(content=final_answer))  # Append clean response to messages
            return state

        except Exception as e:
            error_message = f"Error: {e}"
            state['messages'].append(AIMessage(content=error_message))
            return state

    
    def router_function(self, state: MessagesState) -> Literal["tools", END]:
        """
        Determine the next node based on tool invocation.
        """
        messages = state['messages']
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    def __call__(self):
        """
        Build and return the workflow graph.
        """
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.router_function, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        self.app = workflow.compile()
        return self.app


if __name__ == "__main__":
    mybot = Chatbot()
    workflow = mybot()

    # Properly initialize MessagesState with HumanMessage objects
    initial_state = {
        "messages": [
            HumanMessage(content="Search tthe web and tell me about Airi Shimamura from Oklahoma?")
        ]
    }
    
    response = workflow.invoke(initial_state)
    print(response['messages'][-1].content)
> LangGraph Project 2 (code that works)
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
# from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
import boto3
from typing import Annotated
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


@tool
def compute_savings(monthly_cost: float) -> float:
    """
    Tool to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.
    
    Args:
        monthly_cost (float): The user's current monthly electricity cost.
    
    Returns:
        dict: A dictionary containing:
            - 'number_of_panels': The estimated number of solar panels required.
            - 'installation_cost': The estimated installation cost.
            - 'net_savings_10_years': The net savings over 10 years after installation costs.
    """
    def calculate_solar_savings(monthly_cost):
        # Assumptions for the calculation
        cost_per_kWh = 0.28  
        cost_per_watt = 1.50  
        sunlight_hours_per_day = 3.5  
        panel_wattage = 350  
        system_lifetime_years = 10  
        # Monthly electricity consumption in kWh
        monthly_consumption_kWh = monthly_cost / cost_per_kWh
        
        # Required system size in kW
        daily_energy_production = monthly_consumption_kWh / 30
        system_size_kW = daily_energy_production / sunlight_hours_per_day
        
        # Number of panels and installation cost
        number_of_panels = system_size_kW * 1000 / panel_wattage
        installation_cost = system_size_kW * 1000 * cost_per_watt
        
        # Annual and net savings
        annual_savings = monthly_cost * 12
        total_savings_10_years = annual_savings * system_lifetime_years
        net_savings = total_savings_10_years - installation_cost
        
        return {
            "number_of_panels": round(number_of_panels),
            "installation_cost": round(installation_cost, 2),
            "net_savings_10_years": round(net_savings, 2)
        }
    # Return calculated solar savings
    return calculate_solar_savings(monthly_cost)

def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")
    
    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls
    
    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # Format the error message for the user
                tool_call_id=tc["id"],  # Associate the error message with the corresponding tool call ID
            )
            for tc in tool_calls  # Iterate over each tool call to produce individual error messages
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    
    Args:
        tools (list): A list of tools to be included in the node.
    
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],  # Use a lambda function to wrap the error handler
        exception_key="error"  # Specify that this fallback is for handling errors
    )

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable
    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break
        # Return the final state after processing the runnable
        return {"messages": result}

# def get_bedrock_client(region):
#     return boto3.client("bedrock-runtime", region_name=region)

# def create_bedrock_llm(client):
#     return ChatBedrock(model_id='anthropic.claude-3-sonnet-20240229-v1:0', client=client, model_kwargs={'temperature': 0}, region_name='us-east-1')

# llm = create_bedrock_llm(get_bedrock_client(region='us-east-1'))


llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o",
        streaming=True,
        callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    )

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful customer support assistant for Solar Panels Belgium.
            You should get the following information from them:
            - monthly electricity cost
            If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
            After you are able to discern all the information, call the relevant tool.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)

# Define the tools the assistant will use
part_1_tools = [
    compute_savings
]

# Bind the tools to the assistant's workflow
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools, tool_choice="any")


builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

builder.add_edge(START, "assistant")  # Start with the assistant
builder.add_conditional_edges("assistant", tools_condition)  # Move to tools after input
builder.add_edge("tools", "assistant")  # Return to assistant after tool execution

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# import shutil
import uuid

def _print_event(event, _printed):
    """
    Helper function to print events generated by the graph.
    
    Args:
        event: The event to print.
        _printed: A set to track already printed events to avoid duplicates.
    """
    if event["messages"]:
        for message in event["messages"]:
            if message.id not in _printed:
                # Check the type of the message and print accordingly
                if hasattr(message, "type"):
                    print(f"Type: {message.type}, Content: {message.content}")
                elif hasattr(message, "role"):
                    print(f"Role: {message.role}, Content: {message.content}")
                else:
                    print(f"Message: {message}")
                _printed.add(message.id)
                
                
# Let's create an example conversation a user might have with the assistant
tutorial_questions = [
    'hey',
    'can you calculate my energy saving',
    "my montly cost is $100, what will i save"
]
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "thread_id": thread_id,
    }
}
_printed = set()
for question in tutorial_questions:
    events = graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)

## SMolAgents
> SmolAgents Cheatsheet
# SmoLAGENTS Advanced Cheatsheet

## Table of Contents
    #introduction-to-smolagents)
    #installation-and-setup)
    #agent-types)
    #creating-basic-agents)
    #working-with-tools)
    #creating-custom-tools)
    #using-built-in-tools)
    #loading-tools-from-hub)
    #models-and-integration)
    #multi-agent-systems)
    #advanced-customization)
    #debugging-and-best-practices)
    #security-considerations)
    #examples-and-use-cases)

## Introduction to SmoLAGENTS
    # SmoLAGENTS is a lightweight, minimalist library from Hugging Face designed for creating AI agents with a focus on simplicity and 
    # efficiency. It enables agents to perform actions using Python code snippets rather than JSON or text formats.

    # **Key features:**
    # - Code-based approach to agent actions
    # - Support for various LLM providers
    # - Minimal abstractions (~1000 lines of code for core functionality)
    # - Deep integration with Hugging Face Hub
    # - Support for multi-agent systems
    # - Vision, audio, and other modality support


## Installation and Setup
    # Install the package
    pip install smolagents

    # Basic imports
    from smolagents import CodeAgent, HfApiModel, tool, DuckDuckGoSearchTool


## Agent Types
    # SmoLAGENTS supports two primary agent types:

### 1. CodeAgent
    # CodeAgents generate Python code snippets to perform actions. They're more efficient than traditional tool-calling agents, 
    # using 30% fewer steps and achieving better performance on complex tasks.
    
    from transformers.agents import CodeAgent
    agent = CodeAgent(
        tools=[],  # List of tools the agent can use
        model=HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
        add_base_tools=True  # Adds default tools
    )
    agent.run("What is the result of 2 power 3.7384?")

### 2. ToolCallingAgent
    # ToolCallingAgents generate actions as JSON/text blobs, similar to the approach used by OpenAI and Anthropic.
    from smolagents import ToolCallingAgent
    agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool()],
        model=HfApiModel()
    )
    agent.run("What is the result of 2 power 3.7384?")

### 3. React AGent
    # React Code Agent
        from transformers.agents import ReactCodeAgent
        agent = ReactCodeAgent(
            tools=[DuckDuckGoSearchTool()],
            model=HfApiModel()
        )
        agent.run("What is the result of 2 power 3.7384?")

    # ReacyJson Agent
        from transformers.agents import ReactJsonAgent
        agent = ReactJsonAgent(
            tools=[DuckDuckGoSearchTool()],
            model=HfApiModel()
        )
        agent.run(task = "What is the result of 2 power 3.7384?", stream=True)
    


## Creating Basic Agents

    # 1. **Model** - Powers the agent's reasoning (required)
    # 2. **Tools** - Functions the agent can use (can be empty)


    # Minimal agent creation
    from smolagents import CodeAgent, HfApiModel

    # Choose a model
    model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
    # Or use default model with no token required
    # model = HfApiModel()

    # Create an agent with no tools
    agent = CodeAgent(tools=[], model=model)

    # Run a task
    result = agent.run("Calculate the 10th Fibonacci number")



## Working with Tools
    ### Creating Custom Tools
        #### Method 1: Using the @tool Decorator (Simplest)
            from smolagents import tool
            from typing import Optional

            @tool
            def get_weather(location: str, date: Optional[str] = None) -> str:
                """Gets weather information for a location.
                
                Args:
                    location: The city or place to get weather for
                    date: Optional date in YYYY-MM-DD format, defaults to today
                """
                # Tool implementation here
                import requests
                # API call code
                return "Weather data for location"


        #### Method 2: Creating a Tool Class (More Control)
            from smolagents import Tool

            class HFModelDownloadsTool(Tool):
                name = "model_download_counter"
                description = "Returns the most downloaded model of a given task on the Hugging Face Hub."
                inputs = {
                    "task": {
                        "type": "string",
                        "description": "The task to search for (e.g., 'text-classification', 'image-classification')"
                    }
                }
                output_type = "string"
                
                def forward(self, task: str) -> str:
                    # Implementation here
                    from huggingface_hub import HfApi
                    api = HfApi()
                    # Logic to find most downloaded model
                    return "Model name and download count"

            # or
            
            from transformers import Tool
            from PIL import Image

            class ImageResizeTool(Tool):
                name = "image-resizer"
                description = "Resizes an image to the specified dimensions."
                inputs = {
                    "image": {"type": Image.Image, "description": "The image to resize"},
                    "width": {"type": int, "description": "Target width in pixels"},
                    "height": {"type": int, "description": "Target height in pixels"}
                }
                output_type = Image.Image
                
                def __call__(self, image, width, height):
                    return image.resize((width, height))

    ### Using PipelineTool for Models
        # For tools that wrap Transformer models, you can use the PipelineTool class:
        from transformers import PipelineTool, AutoModelForCausalLM, AutoTokenizer

        class TextGeneratorTool(PipelineTool):
            name = "text-generator"
            description = "Generates text based on a prompt."
            default_checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            model_class = AutoModelForCausalLM
            pre_processor_class = AutoTokenizer
            
            def __call__(self, prompt: str, max_length: int = 100) -> str:
                """
                Generates text based on the input prompt.
                
                Args:
                    prompt: The text prompt to generate from.
                    max_length: Maximum length of generated text.
                    
                Returns:
                    Generated text.
                """
                inputs = self.encode(prompt)
                outputs = self.forward(inputs)
                return self.decode(outputs)[0]["generated_text"]


    ### Using Built-in Tools
        # SmoLAGENTS provides a default toolbox that you can add with `add_base_tools=True`:

            # Default tools include:
            # - DuckDuckGo web search
            # - Python code interpreter (for ToolCallingAgent)
            # - Transcriber (speech-to-text)

            # Add just the DuckDuckGo search tool explicitly
            from smolagents import DuckDuckGoSearchTool
            agent = CodeAgent(
                tools=[DuckDuckGoSearchTool()], 
                model=HfApiModel()
            )

            # Using a tool directly
            search_tool = DuckDuckGoSearchTool()
            result = search_tool("What is the capital of France?")
            print(result)


    ### Loading Tools from Hub
        from smolagents import load_tool, CodeAgent, HfApiModel

        # Load a tool from the Hub (requires trust_remote_code=True for security)
        image_generation_tool = load_tool(
            "m-ric/text-to-image", 
            trust_remote_code=True
        )

        # Use the loaded tool in an agent
        agent = CodeAgent(
            tools=[image_generation_tool],
            model=HfApiModel()
        )


    #### Loading a Space as a Tool
        from smolagents import Tool

        # Import a Hugging Face Space as a tool
        image_generation_tool = Tool.from_space(
            "black-forest-labs/FLUX.1-schnell",
            name="image_generator",
            description="Generate an image from a prompt"
        )

        # or
    
        from transformers import Tool

        image_generator = Tool.from_space(
            space_id="stabilityai/stable-diffusion",
            name="image-generator",
            description="Generates images from text prompts using Stable Diffusion."
        )


    #### From other frameworks

        # From LangChain
        from transformers import Tool
        from langchain.tools import BaseTool

        class LangChainCalculatorTool(BaseTool):
            name = "calculator"
            description = "Useful for performing calculations"
            
            def _run(self, query):
                return eval(query)
            
            def _arun(self, query):
                return eval(query)

        calculator_tool = Tool.from_langchain(LangChainCalculatorTool())

        # From Gradio
        import gradio as gr
        from transformers import Tool

        def calculator_fn(expression):
            return eval(expression)

        with gr.Blocks() as demo:
            inp = gr.Textbox(label="Expression")
            out = gr.Number(label="Result")
            btn = gr.Button("Calculate")
            btn.click(fn=calculator_fn, inputs=inp, outputs=out)

        calculator_tool = Tool.from_gradio(demo)



## Models and Integration
        # SmoLAGENTS is model-agnostic and supports various LLM providers:


    ### HfApiModel (Hugging Face)
        from smolagents import HfApiModel

        # Default free model
        model = HfApiModel()

        # Specific Hugging Face model
        model = HfApiModel(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            token="YOUR_HF_TOKEN"  # Optional
        )

        # Using a specific provider
        model = HfApiModel(
            model_id="deepseek-ai/DeepSeek-R1",
            provider="together",
        )


    ### LiteLLMModel (Multiple Providers)
        import os
        from smolagents import LiteLLMModel

        # Access models from various providers
        model = LiteLLMModel(
            model_id="anthropic/claude-3-5-sonnet-latest",
            temperature=0.2,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )


    ### OpenAIServerModel
        import os
        from smolagents import OpenAIServerModel

        model = OpenAIServerModel(
            model_id="deepseek-ai/DeepSeek-R1",
            api_base="https://api.together.xyz/v1/",  # Leave empty for OpenAI
            api_key=os.environ["TOGETHER_API_KEY"]
        )




## Multi-Agent Systems
    # SmoLAGENTS supports hierarchical multi-agent systems for specialized tasks:
        from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool

        # Create specialized agents
        model = HfApiModel()

        # Web search agent
        web_agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],
            model=model,
            name="web_search",
            description="Runs web searches for you. Give it your query as an argument."
        )

        # Image generation agent
        image_agent = CodeAgent(
            tools=[image_generation_tool],  # From previous examples
            model=model,
            name="image_creator",
            description="Creates images based on prompts. Provide a detailed description of the image."
        )

        # Manager agent that coordinates other agents
        manager_agent = CodeAgent(
            tools=[],  # No direct tools
            model=model,
            managed_agents=[web_agent, image_agent]  # Manages other agents
        )

        # Run a complex task that requires both web search and image generation
        manager_agent.run("Research current fashion trends and create an image of a modern business outfit")






## Advanced Customization

    ### Custom System Prompts

        from smolagents import CodeAgent, HfApiModel
        from smolagents.prompts import CODE_SYSTEM_PROMPT

        # Customize the system prompt
        modified_system_prompt = CODE_SYSTEM_PROMPT + "\nAlways explain your reasoning step by step."

        # Use the modified prompt
        agent = CodeAgent(
            tools=[],
            model=HfApiModel(),
            system_prompt=modified_system_prompt
        )



    ### Passing Additional Arguments

        from smolagents import CodeAgent, HfApiModel

        agent = CodeAgent(tools=[], model=HfApiModel())

        # Pass additional data to the agent
        result = agent.run(
            "Debug this code and fix any errors",
            additional_args={
                "code": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
            }
        )


## Debugging and Best Practices

    ### Logging and Inspection

    # Run an agent
    agent.run("Your task here")

    # Access logs to inspect what happened
    print(agent.logs)

    # Get a higher-level view of the agent's memory as chat messages
    messages = agent.write_memory_to_messages()
    print(messages)


## Security Considerations
    # Code execution brings security concerns. SmoLAGENTS offers several security options:

    # 1. **Default Local Execution**: Limited to provided tools and safe functions
    # 2. **E2B Sandboxed Environment**: For safer code execution
    # 3. **Docker Sandboxing**: Isolate code execution in containers


    # Using E2B for sandboxed code execution
    from smolagents import CodeAgent, HfApiModel
    from smolagents.execution import E2BExecutor

    # Create an E2B executor (requires E2B API key)
    executor = E2BExecutor()

    # Create agent with sandboxed execution
    agent = CodeAgent(
        tools=[],
        model=HfApiModel(),
        executor=executor
    )



> SmolAgents Examples 1
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool

# Check the name of the search tool
search_tool = DuckDuckGoSearchTool()
print(f"Search tool name: {search_tool.name}")  # See what name this tool has

# Create a model
model = HfApiModel()

# Create a specialized agent with a different name
web_search_agent = CodeAgent(
    tools=[search_tool],
    model=model,
    name="web_search_agent",  # Changed from 'web_search' to 'web_search_agent'
    description="Runs web searches for you. Give it your query as an argument."
)

# Create a manager agent
manager_agent = CodeAgent(
    tools=[],  # No direct tools
    model=model,
    managed_agents=[web_search_agent]
)

# Run a task with the manager
result = manager_agent.run("Who is the CEO of Hugging Face?")
print(result)
> > SmolAgents Examples 2
import os
from typing import Optional, List, Dict, Any
import json
import random
import datetime
from datetime import timedelta

# Import SmoLAGENTS components
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, tool

class TravelPlanningSystem:
    """
    Advanced Travel Planning System with multiple specialized agents.
    This system demonstrates a comprehensive multi-agent setup using SmoLAGENTS.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the travel planning system with multiple specialized agents."""
        # Setup model with token if provided
        self.model = HfApiModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            token=hf_token
        )
        
        # Create specialized tools
        self.setup_tools()
        
        # Create specialized agents
        self.setup_agents()
        
        # Initialize conversation history
        self.history = []
    
    def setup_tools(self):
        """Set up specialized tools for each agent."""
        # Web search tool for general information
        self.search_tool = DuckDuckGoSearchTool()
        
        # Weather information tool (mock implementation)
        @tool
        def get_weather_forecast(location: str, date: Optional[str] = None) -> str:
            """Get weather forecast for a specific location and date.
            
            Args:
                location: City or location
                date: Date in YYYY-MM-DD format (default: today)
            """
            # Mock implementation with consistent randomization
            weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorms", "Snowy"]
            temperatures = {
                "Sunny": (75, 95),
                "Partly Cloudy": (70, 85),
                "Cloudy": (65, 80),
                "Light Rain": (60, 75),
                "Heavy Rain": (55, 70),
                "Thunderstorms": (50, 65),
                "Snowy": (25, 40)
            }
            
            # Use location and date as random seed for consistent results
            if date is None:
                date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Generate seed based on location and date
            seed = hash(f"{location}-{date}")
            random.seed(seed)
            
            # Select weather condition and temperature
            condition = random.choice(weather_conditions)
            temp_range = temperatures[condition]
            temp = random.randint(temp_range[0], temp_range[1])
            
            return f"Weather forecast for {location} on {date}: {condition}, {temp}°F"
        
        self.weather_tool = get_weather_forecast
        
        # Flight information tool (mock implementation)
        @tool
        def search_flights(origin: str, destination: str, date: str) -> str:
            """Search for flight options between cities.
            
            Args:
                origin: Departure city or airport
                destination: Arrival city or airport
                date: Travel date in YYYY-MM-DD format
            """
            # Mock implementation
            seed = hash(f"{origin}-{destination}-{date}")
            random.seed(seed)
            
            airlines = ["AirGlobal", "SkyWays", "TransWorld", "StarLines", "OceanicAir"]
            flight_count = random.randint(3, 5)
            flights = []
            
            for i in range(flight_count):
                airline = random.choice(airlines)
                flight_num = f"{airline[:2].upper()}{random.randint(100, 999)}"
                departure_hour = random.randint(6, 22)
                flight_duration = random.randint(2, 8)
                arrival_hour = (departure_hour + flight_duration) % 24
                price = random.randint(250, 1200)
                
                flights.append({
                    "airline": airline,
                    "flight": flight_num,
                    "departure": f"{date} {departure_hour:02d}:00",
                    "arrival": f"{date} {arrival_hour:02d}:{random.randint(0, 59):02d}",
                    "duration": f"{flight_duration}h {random.randint(0, 59)}m",
                    "price": f"${price}"
                })
            
            # Format flight information
            result = f"Found {len(flights)} flights from {origin} to {destination} on {date}:\n\n"
            for i, flight in enumerate(flights, 1):
                result += f"Option {i}:\n"
                result += f"  Airline: {flight['airline']}\n"
                result += f"  Flight: {flight['flight']}\n"
                result += f"  Departure: {flight['departure']}\n"
                result += f"  Arrival: {flight['arrival']}\n"
                result += f"  Duration: {flight['duration']}\n"
                result += f"  Price: {flight['price']}\n\n"
            
            return result
        
        self.flight_tool = search_flights
        
        # Hotel information tool (mock implementation)
        @tool
        def search_hotels(location: str, check_in: str, check_out: str, budget: Optional[str] = "mid") -> str:
            """Search for hotel options in a location.
            
            Args:
                location: City or area
                check_in: Check-in date in YYYY-MM-DD format
                check_out: Check-out date in YYYY-MM-DD format
                budget: "budget", "mid", or "luxury" (default: "mid")
            """
            # Mock implementation
            seed = hash(f"{location}-{check_in}-{budget}")
            random.seed(seed)
            
            hotel_prefixes = ["Grand", "Royal", "Luxury", "Comfort", "Premier", "Elite", "Plaza"]
            hotel_suffixes = ["Hotel", "Inn", "Suites", "Resort", "Lodge", "Place", "Residences"]
            hotel_areas = ["Downtown", "City Center", "Riverside", "Beachfront", "Old Town", "Business District"]
            
            # Price ranges based on budget
            price_ranges = {
                "budget": (50, 120),
                "mid": (130, 300),
                "luxury": (320, 800)
            }
            
            # Generate hotels
            hotel_count = random.randint(3, 6)
            hotels = []
            
            for _ in range(hotel_count):
                name = f"{random.choice(hotel_prefixes)} {location} {random.choice(hotel_suffixes)}"
                area = random.choice(hotel_areas)
                price_range = price_ranges.get(budget.lower(), price_ranges["mid"])
                price = random.randint(price_range[0], price_range[1])
                rating = round(random.uniform(3.0, 5.0), 1)
                
                hotels.append({
                    "name": name,
                    "area": area,
                    "price_per_night": f"${price}",
                    "rating": rating,
                    "amenities": random.sample(["WiFi", "Pool", "Gym", "Restaurant", "Spa", "Room Service"], k=random.randint(2, 5))
                })
            
            # Format hotel information
            result = f"Found {len(hotels)} hotels in {location} from {check_in} to {check_out} ({budget} budget):\n\n"
            for i, hotel in enumerate(hotels, 1):
                result += f"Option {i}: {hotel['name']}\n"
                result += f"  Location: {hotel['area']}\n"
                result += f"  Price: {hotel['price_per_night']} per night\n"
                result += f"  Rating: {hotel['rating']}/5.0\n"
                result += f"  Amenities: {', '.join(hotel['amenities'])}\n\n"
            
            return result
        
        self.hotel_tool = search_hotels
        
        # Attraction recommendation tool
        @tool
        def recommend_attractions(location: str, interests: Optional[str] = "general") -> str:
            """Recommend tourist attractions in a location based on interests.
            
            Args:
                location: City or area
                interests: Comma-separated list of interests (e.g., "history,food,nature")
            """
            # Use search for real recommendations, then format nicely
            search_query = f"top tourist attractions in {location} for {interests} travelers"
            results = self.search_tool(search_query)
            
            return f"Recommended attractions in {location} for {interests} interests:\n\n{results}"
        
        self.attraction_tool = recommend_attractions
        
        # Currency converter tool (mock implementation)
        @tool
        def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
            """Convert currency from one type to another.
            
            Args:
                amount: Amount to convert
                from_currency: Source currency code (e.g., USD, EUR, GBP)
                to_currency: Target currency code (e.g., USD, EUR, GBP)
            """
            # Mock exchange rates
            rates = {
                "USD": 1.0,
                "EUR": 0.92,
                "GBP": 0.78,
                "JPY": 150.2,
                "CAD": 1.35,
                "AUD": 1.51,
                "CNY": 7.21,
                "INR": 83.12,
            }
            
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()
            
            if from_currency not in rates:
                return f"Currency {from_currency} not supported. Supported currencies: {', '.join(rates.keys())}"
            
            if to_currency not in rates:
                return f"Currency {to_currency} not supported. Supported currencies: {', '.join(rates.keys())}"
            
            # Convert to USD first, then to target currency
            usd_amount = amount / rates[from_currency]
            target_amount = usd_amount * rates[to_currency]
            
            return f"{amount} {from_currency} = {target_amount:.2f} {to_currency}"
        
        self.currency_tool = convert_currency
        
        # Create itinerary tool
        @tool
        def create_daily_itinerary(location: str, date: str, start_time: str, end_time: str, 
                                 interests: Optional[str] = "general") -> str:
            """Create a daily itinerary for a specific location and date.
            
            Args:
                location: City or area
                date: Date in YYYY-MM-DD format
                start_time: Start time in HH:MM format (24-hour)
                end_time: End time in HH:MM format (24-hour)
                interests: Comma-separated list of interests (e.g., "history,food,nature")
            """
            # Use search to get recommendations
            search_query = f"one day itinerary {location} {interests} attractions"
            search_results = self.search_tool(search_query)
            
            # Format as itinerary
            # Parse start and end times
            start_hour, start_minute = map(int, start_time.split(':'))
            end_hour, end_minute = map(int, end_time.split(':'))
            
            # Create time slots
            current_hour, current_minute = start_hour, start_minute
            end_time_minutes = end_hour * 60 + end_minute
            itinerary = f"Daily Itinerary for {location} on {date} ({interests}):\n\n"
            
            # Mock itinerary based on search results
            remaining_time_minutes = end_time_minutes - (current_hour * 60 + current_minute)
            
            if remaining_time_minutes <= 0:
                return "Invalid time range. End time must be after start time."
            
            # Add weather information
            itinerary += f"Weather: {self.weather_tool(location, date)}\n\n"
            
            # Seed random generator for consistent results
            random.seed(hash(f"{location}-{date}-{interests}"))
            
            # Create itinerary
            while remaining_time_minutes > 0:
                activity_duration = random.randint(1, 3) * 60  # 1-3 hours in minutes
                
                # Don't go past end time
                if current_hour * 60 + current_minute + activity_duration > end_time_minutes:
                    activity_duration = end_time_minutes - (current_hour * 60 + current_minute)
                    
                if activity_duration <= 0:
                    break
                
                # Format time slot
                time_slot = f"{current_hour:02d}:{current_minute:02d}"
                
                # Update time
                current_minute += activity_duration
                while current_minute >= 60:
                    current_hour += 1
                    current_minute -= 60
                
                end_slot = f"{current_hour:02d}:{current_minute:02d}"
                
                # Add activity to itinerary
                itinerary += f"{time_slot} - {end_slot}: Activity based on {interests} interests\n"
                
                # Add travel/buffer time
                buffer_time = random.randint(15, 45)  # 15-45 minutes
                if current_minute + buffer_time >= 60:
                    current_hour += (current_minute + buffer_time) // 60
                    current_minute = (current_minute + buffer_time) % 60
                else:
                    current_minute += buffer_time
                
                remaining_time_minutes = end_time_minutes - (current_hour * 60 + current_minute)
                
                # Add meals at appropriate times
                if (8 <= current_hour < 10 and random.random() < 0.7):
                    itinerary += f"{current_hour:02d}:{current_minute:02d} - {current_hour+1:02d}:{current_minute:02d}: Breakfast\n"
                    current_hour += 1
                elif (12 <= current_hour < 14 and random.random() < 0.7):
                    itinerary += f"{current_hour:02d}:{current_minute:02d} - {current_hour+1:02d}:{current_minute:02d}: Lunch\n"
                    current_hour += 1
                elif (18 <= current_hour < 20 and random.random() < 0.7):
                    itinerary += f"{current_hour:02d}:{current_minute:02d} - {current_hour+1:02d}:{current_minute:02d}: Dinner\n"
                    current_hour += 1
                
                remaining_time_minutes = end_time_minutes - (current_hour * 60 + current_minute)
            
            itinerary += f"\nRecommendations based on web search:\n{search_results}"
            
            return itinerary
        
        self.itinerary_tool = create_daily_itinerary
    
    def setup_agents(self):
        """Set up specialized agents for different travel planning tasks."""
        # Create the information search agent
        self.info_agent = CodeAgent(
            tools=[self.search_tool],
            model=self.model,
            name="info_agent",
            description="Searches for general information on destinations, travel requirements, etc."
        )
        
        # Create the weather agent
        self.weather_agent = CodeAgent(
            tools=[self.weather_tool],
            model=self.model,
            name="weather_agent",
            description="Provides weather forecasts for specific destinations and dates."
        )
        
        # Create the flight booking agent
        self.flight_agent = CodeAgent(
            tools=[self.flight_tool, self.search_tool],
            model=self.model,
            name="flight_agent",
            description="Searches for flight options between destinations."
        )
        
        # Create the accommodation agent
        self.hotel_agent = CodeAgent(
            tools=[self.hotel_tool, self.search_tool],
            model=self.model,
            name="hotel_agent",
            description="Searches for hotel and accommodation options in destinations."
        )
        
        # Create the attractions agent
        self.attraction_agent = CodeAgent(
            tools=[self.attraction_tool, self.search_tool],
            model=self.model,
            name="attraction_agent",
            description="Recommends tourist attractions and activities in destinations."
        )
        
        # Create the currency agent
        self.currency_agent = CodeAgent(
            tools=[self.currency_tool],
            model=self.model,
            name="currency_agent",
            description="Converts currencies to help with budget planning."
        )
        
        # Create the itinerary creation agent
        self.itinerary_agent = CodeAgent(
            tools=[self.itinerary_tool, self.weather_tool, self.search_tool],
            model=self.model,
            name="itinerary_agent",
            description="Creates detailed daily itineraries for destinations."
        )
        
        # Create the travel planner agent (orchestrator)
        self.planner_agent = CodeAgent(
            tools=[],  # No direct tools
            model=self.model,
            managed_agents=[
                self.info_agent,
                self.weather_agent,
                self.flight_agent,
                self.hotel_agent,
                self.attraction_agent,
                self.currency_agent,
                self.itinerary_agent
            ],
            name="travel_planner",
            description="Plans comprehensive travel itineraries by coordinating specialized agents."
        )
    
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def process_query(self, query: str) -> str:
        """Process a user query through the travel planning system."""
        # Add user message to history
        self.add_to_history("user", query)
        
        # Use the planner agent to process the query
        result = self.planner_agent.run(
            query,
            additional_args={"conversation_history": self.history}
        )
        
        # Add the response to history
        self.add_to_history("assistant", result)
        
        return result

# Example usage
def main():
    # Initialize the travel planning system
    system = TravelPlanningSystem()
    
    # Example queries
    queries = [
        "I'm planning a trip to Tokyo in April 2025. What's the weather like and what are the must-see attractions?",
        "Can you find flights from New York to Paris on June 15, 2025 and hotels for a 5-night stay?",
        "Create a one-day itinerary for London focusing on historical sites. Start at 9:00 and end at 18:00.",
        "How much is 500 USD in Japanese Yen?",
        "I'm planning a 3-day trip to Barcelona. Can you suggest an itinerary including flights, hotels, and attractions?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nUser: {query}")
        response = system.process_query(query)
        print(f"\nTravel Planning System: {response}")
        print("\n" + "="*80)
    
    # Interactive mode
    print("\nEnter your travel questions (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        
        response = system.process_query(query)
        print(f"\nTravel Planning System: {response}")

if __name__ == "__main__":
    main()

## RAG
> ColBert
# ColBERT (Contextualized Late Interaction using BERT) is an innovative approach to Retrieval Augmented Generation (RAG) that overcomes 
# limitations of traditional Dense Passage Retrieval (DPR). Unlike DPR, which represents passages as single vectors, ColBERT generates 
# multiple contextual vectors for each token in passages and queries. Relevance scoring is calculated by summing the maximum similarity 
# between each query token and any document token. This method significantly improves performance on unusual terms and names while reducing 
# sensitivity to chunking strategies. ColBERT outperforms more complex solutions on benchmark datasets with only a slight increase in 
# latency, making it an effective enhancement for RAG applications.


import torch

def compute_relevance_scores(query_embeddings, document_embeddings, k):
    """
    Compute relevance scores for top-k documents given a query.
    
    :param query_embeddings: Tensor representing the query embeddings, shape: [num_query_terms, embedding_dim]
    :param document_embeddings: Tensor representing embeddings for k documents, shape: [k, max_doc_length, embedding_dim]
    :param k: Number of top documents to re-rank
    :return: Sorted document indices based on their relevance scores
    """
    
    # Ensure document_embeddings is a 3D tensor: [k, max_doc_length, embedding_dim]
    # Pad the k documents to their maximum length for batch operations
    # Note: Assuming document_embeddings is already padded and moved to GPU
    
    # Compute batch dot-product of Eq (query embeddings) and D (document embeddings)
    # Resulting shape: [k, num_query_terms, max_doc_length]
    scores = torch.matmul(query_embeddings.unsqueeze(0), document_embeddings.transpose(1, 2))
    
    print("scores_shape", scores.shape)
    # Apply max-pooling across document terms (dim=2) to find the max similarity per query term
    # Shape after max-pool: [k, num_query_terms]
    max_scores_per_query_term = scores.max(dim=2).values
    print("max_scores_per_query_term_shape", max_scores_per_query_term.shape)
    # Sum the scores across query terms to get the total score for each document
    # Shape after sum: [k]
    total_scores = max_scores_per_query_term.sum(dim=1)
    print("total_scores", total_scores)
    # Sort the documents based on their total scores
    sorted_indices = total_scores.argsort(descending=True)
    
    return sorted_indices

def test_compute_relevance_scores():
    # Set dimensions
    num_query_terms = 3  # number of tokens in query
    embedding_dim = 5    # dimension of each embedding
    k = 7               # number of documents to rerank
    max_doc_length = 4  # example document length
    
    # Create sample query embeddings: shape [3, 5]
    query_embeddings = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # embedding for first query token
        [0.2, 0.3, 0.4, 0.5, 0.6],  # embedding for second query token
        [0.3, 0.4, 0.5, 0.6, 0.7]   # embedding for third query token
    ])
    
    # Create sample document embeddings: shape [7, 4, 5]
    document_embeddings = torch.randn(k, max_doc_length, embedding_dim)
    
    # Compute relevance scores
    sorted_indices = compute_relevance_scores(query_embeddings, document_embeddings, k)
    
    # Test assertions
    assert sorted_indices.shape == torch.Size([k]), "Output shape should be [k]"
    assert len(torch.unique(sorted_indices)) == k, "All indices should be unique"
    assert all(0 <= idx < k for idx in sorted_indices), "Indices should be in range [0, k)"
    
    print("Test passed successfully!")
    print("Sorted indices:", sorted_indices.tolist())

# Run the test
test_compute_relevance_scores()

# scores_shape torch.Size([7, 3, 4])
# max_scores_per_query_term_shape torch.Size([7, 3])
# total_scores tensor([-0.1476,  0.7772,  2.1757,  3.3793,  4.8741,  4.0813,  1.8585])
# Test passed successfully!
# Sorted indices: [4, 5, 3, 2, 6, 1, 0]