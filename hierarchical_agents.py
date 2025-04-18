"""
Hierarchical Multi-Agent Architecture

A structured tree-like organization where a main agent delegates to subordinate agents.
This allows for complex task decomposition with specialized agents handling subtasks.
"""

import functools
import asyncio
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Literal, Optional
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from utils import get_model, get_common_tools, create_agent_prompt
from utils import RESEARCHER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT, PLANNER_PROMPT

# Define the state for the hierarchical agents workflow
class HierarchicalAgentState(TypedDict):
    """State for hierarchical agents workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    subtasks: List[Dict[str, Any]]
    current_subtask: Optional[Dict[str, Any]]
    results: Dict[str, Any]
    current_team: str
    current_agent: str
    is_complete: bool

# Define specialized prompts for different roles
SUPERVISOR_PROMPT = """You are a supervisor agent responsible for coordinating multiple specialized teams.
Your role is to analyze complex tasks, break them down into subtasks, assign them to appropriate teams,
and synthesize the results into a cohesive final response.

You have the following teams at your disposal:
- RESEARCH_TEAM: For information gathering, fact-finding, and data collection
- ANALYSIS_TEAM: For data processing, interpretation, and insight generation
- CONTENT_TEAM: For content creation, writing, editing, and presentation

Approach each task methodically by:
1. Breaking it down into well-defined subtasks
2. Assigning each subtask to the most appropriate team
3. Reviewing team outputs and providing feedback if necessary
4. Synthesizing all contributions into a comprehensive final response"""

RESEARCH_LEAD_PROMPT = """You are the research team lead responsible for coordinating information gathering efforts.
When assigned a research subtask, you'll determine what specific research operations are needed and
delegate them to your specialized research agents."""

ANALYSIS_LEAD_PROMPT = """You are the analysis team lead responsible for coordinating data processing and interpretation.
When assigned an analysis subtask, you'll determine what specific analytical operations are needed and
delegate them to your specialized analysis agents."""

CONTENT_LEAD_PROMPT = """You are the content team lead responsible for coordinating content creation and presentation.
When assigned a content subtask, you'll determine what specific content operations are needed and
delegate them to your specialized content agents."""

# Research team members
WEB_SEARCH_PROMPT = """You are a web search specialist focusing on finding current and relevant information
online related to the assigned research task."""

KNOWLEDGE_BASE_PROMPT = """You are a knowledge base specialist with deep expertise across various domains.
Your role is to provide factual and comprehensive information from your internal knowledge."""

# Analysis team members
DATA_ANALYST_PROMPT = """You are a data analyst specializing in processing and interpreting numerical and 
statistical information to identify patterns, trends, and insights relevant to the task."""

CRITICAL_THINKER_PROMPT = """You are a critical thinking specialist focusing on logical analysis, identification
of assumptions, evaluation of arguments, and spotting potential biases or fallacies."""

# Content team members
WRITER_SPECIALIZED_PROMPT = """You are a specialized content writer focusing on creating clear, engaging, and
well-structured text that effectively communicates complex information."""

EDITOR_PROMPT = """You are an editing specialist focusing on improving clarity, flow, grammar, and overall
quality of content while ensuring it meets the objectives of the assigned task."""

# Node functions

def supervisor_node(state: HierarchicalAgentState) -> Dict:
    """Main supervisor agent that coordinates the overall workflow."""
    if not state.get("task"):
        task = state["messages"][0].content
        llm = get_model()

        # --- WORKAROUND: Manual String Construction ---
        prompt_text = f"""You are a task management supervisor responsible for breaking down complex tasks
            into smaller, manageable subtasks and assigning them to specialized teams.

            Task: {task}

            Break this task down into 3-5 subtasks, each suited for one of these teams:
            - RESEARCH_TEAM: For information gathering, fact-finding, and data collection
            - ANALYSIS_TEAM: For data processing, interpretation, and insight generation
            - CONTENT_TEAM: For content creation, writing, editing, and presentation

            For each subtask, provide:
            1. A clear, specific description
            2. The team it should be assigned to
            3. Priority order (1 being highest)

            Format your response as a JSON-like structure with subtasks, but in plain text:
            [
              {{"description": "subtask 1 description", "team": "TEAM_NAME", "priority": 1}},
              {{"description": "subtask 2 description", "team": "TEAM_NAME", "priority": 2}},
              ...
            ]
            """
        # Note: Using f-string here automatically handles the {task} substitution.
        # We use {{ and }} now because the f-string itself needs to escape the literal braces.

        print("\n--- DEBUG: Using manually constructed prompt ---")
        # print(prompt_text) # Optional: Print the final prompt

        # Invoke the LLM directly with the constructed string
        response = llm.invoke(prompt_text)
        # --- END WORKAROUND ---

        # ... rest of the subtask parsing logic ...
        subtasks_text = response.content
        import re
        import json
        # ... (parsing logic remains the same) ...

        # Default subtasks if parsing fails
        subtasks = [
            {"description": "Research current information on the topic", "team": "RESEARCH_TEAM", "priority": 1},
            {"description": "Analyze key aspects and implications", "team": "ANALYSIS_TEAM", "priority": 2},
            {"description": "Create comprehensive content addressing the task", "team": "CONTENT_TEAM", "priority": 3}
        ]

        # ... (rest of parsing and return logic) ...
        match = re.search(r'\[(.*?)\]', subtasks_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                json_str = json_str.replace("'", '"')
                json_str = f"[{json_str}]"
                json_str = re.sub(r'\s+', ' ', json_str)
                parsed_subtasks = json.loads(json_str)
                valid_subtasks = []
                for st in parsed_subtasks:
                    if isinstance(st, dict) and "description" in st and "team" in st and "priority" in st:
                        valid_subtasks.append(st)
                if valid_subtasks:
                    subtasks = valid_subtasks
            except Exception as e:
                print(f"JSON parsing error: {e}")

        results = {}
        return {
            "task": task,
            "subtasks": sorted(subtasks, key=lambda x: x.get("priority", 999)),
            "results": results,
            "is_complete": False
        }

    # ... (rest of supervisor_node for handling completion) ...
    elif state.get("subtasks") and not state.get("is_complete"):
         # ... (logic remains the same) ...
         all_complete = True
         for subtask in state["subtasks"]:
             if subtask["description"] not in state["results"]:
                 all_complete = False
                 break

         if all_complete:
             task = state["task"]
             results = state["results"]
             llm = get_model()
             finalize_prompt = PromptTemplate(
                  template="""You are a synthesis specialist responsible for creating a comprehensive final response
                  based on the work of multiple specialized teams.

                  Original Task: {task}

                  Results from teams:
                  {results}

                  Create a comprehensive, well-structured final response that:
                  1. Directly addresses the original task
                  2. Integrates all the information and insights from the different teams
                  3. Is cohesive, clear, and well-organized
                  4. Provides a complete solution or answer to the task
                  """,
                  input_variables=["task", "results"]
             )
             results_formatted = ""
             for subtask, result in results.items():
                  results_formatted += f"\nSubtask: {subtask}\nResult: {result}\n"
             response = llm.invoke(finalize_prompt.format(task=task, results=results_formatted))
             return {
                  "messages": [AIMessage(content=response.content, name="supervisor")],
                  "is_complete": True
             }
         else:
             for subtask in state["subtasks"]:
                 if subtask["description"] not in state["results"]:
                     return {
                         "current_subtask": subtask,
                         "current_team": subtask["team"]
                     }

    return {} # Default return

# Team lead nodes
def research_lead_node(state: HierarchicalAgentState) -> Dict:
    """Research team lead that delegates to specialized research agents."""
    # Get the current subtask
    subtask = state["current_subtask"]
    
    # Create the research lead agent
    llm = get_model()
    lead_prompt = PromptTemplate(
        template="""You are the research team lead. Your job is to break down a research task
        into specific research operations and assign them to specialized agents.
        
        Research Task: {task}
        Original Task Context: {original_task}
        
        Break this down into 2 specific research operations:
        1. One focusing on web search to find current information
        2. One focusing on established knowledge and background information
        
        Format your response as a JSON-like structure with operations, but in plain text:
        [
          {{"operation": "operation 1 description", "agent": "web_search"}},
          {{"operation": "operation 2 description", "agent": "knowledge_base"}}
        ]
        """,
        input_variables=["task", "original_task"]
    )
    
    # Generate the research operations
    response = llm.invoke(lead_prompt.format(
        task=subtask["description"],
        original_task=state["task"]
    ))
    
    # Parse the operations - in a real implementation, you would use proper parsing
    import re
    import json
    
    operations = []
    # Find text between square brackets
    match = re.search(r'\[(.*?)\]', response.content, re.DOTALL)
    if match:
        try:
            # Try to parse as JSON
            operations_json = f"[{match.group(1)}]"
            operations = json.loads(operations_json)
        except:
            # Fallback to manual parsing if JSON parse fails
            operations = [
                {"operation": f"Web search: {subtask['description']}", "agent": "web_search"},
                {"operation": f"Knowledge base search: {subtask['description']}", "agent": "knowledge_base"}
            ]
    else:
        # Default operations if parsing fails
        operations = [
            {"operation": f"Web search: {subtask['description']}", "agent": "web_search"},
            {"operation": f"Knowledge base search: {subtask['description']}", "agent": "knowledge_base"}
        ]
    
    return {
        # "current_team": "RESEARCH_TEAM",
        "current_agent": operations[0]["agent"],
        "operations": operations
    }

def analysis_lead_node(state: HierarchicalAgentState) -> Dict:
    """Analysis team lead that delegates to specialized analysis agents."""
    # Get the current subtask
    subtask = state["current_subtask"]
    
    # Create the analysis lead agent
    llm = get_model()
    lead_prompt = PromptTemplate(
        template="""You are the analysis team lead. Your job is to break down an analysis task
        into specific analytical operations and assign them to specialized agents.
        
        Analysis Task: {task}
        Original Task Context: {original_task}
        
        Break this down into 2 specific analytical operations:
        1. One focusing on data analysis and pattern identification
        2. One focusing on critical thinking and logical evaluation
        
        Format your response as a JSON-like structure with operations, but in plain text:
        [
          {{"operation": "operation 1 description", "agent": "data_analyst"}},
          {{"operation": "operation 2 description", "agent": "critical_thinker"}}
        ]
        """,
        input_variables=["task", "original_task"]
    )
    
    # Generate the analysis operations
    response = llm.invoke(lead_prompt.format(
        task=subtask["description"],
        original_task=state["task"]
    ))
    
    # Parse the operations - in a real implementation, you would use proper parsing
    import re
    import json
    
    operations = []
    # Find text between square brackets
    match = re.search(r'\[(.*?)\]', response.content, re.DOTALL)
    if match:
        try:
            # Try to parse as JSON
            operations_json = f"[{match.group(1)}]"
            operations = json.loads(operations_json)
        except:
            # Fallback to manual parsing if JSON parse fails
            operations = [
                {"operation": f"Data analysis: {subtask['description']}", "agent": "data_analyst"},
                {"operation": f"Critical thinking: {subtask['description']}", "agent": "critical_thinker"}
            ]
    else:
        # Default operations if parsing fails
        operations = [
            {"operation": f"Data analysis: {subtask['description']}", "agent": "data_analyst"},
            {"operation": f"Critical thinking: {subtask['description']}", "agent": "critical_thinker"}
        ]
    
    return {
        # "current_team": "ANALYSIS_TEAM",
        "current_agent": operations[0]["agent"],
        "operations": operations
    }

def content_lead_node(state: HierarchicalAgentState) -> Dict:
    """Content team lead that delegates to specialized content agents."""
    # Get the current subtask
    subtask = state["current_subtask"]
    
    # Create the content lead agent
    llm = get_model()
    lead_prompt = PromptTemplate(
        template="""You are the content team lead. Your job is to break down a content creation task
        into specific content operations and assign them to specialized agents.
        
        Content Task: {task}
        Original Task Context: {original_task}
        
        Break this down into 2 specific content operations:
        1. One focusing on writing initial content
        2. One focusing on editing and refining the content
        
        Format your response as a JSON-like structure with operations, but in plain text:
        [
          {{"operation": "operation 1 description", "agent": "writer"}},
          {{"operation": "operation 2 description", "agent": "editor"}}
        ]
        """,
        input_variables=["task", "original_task"]
    )
    
    # Generate the content operations
    response = llm.invoke(lead_prompt.format(
        task=subtask["description"],
        original_task=state["task"]
    ))
    
    # Parse the operations - in a real implementation, you would use proper parsing
    import re
    import json
    
    operations = []
    # Find text between square brackets
    match = re.search(r'\[(.*?)\]', response.content, re.DOTALL)
    if match:
        try:
            # Try to parse as JSON
            operations_json = f"[{match.group(1)}]"
            operations = json.loads(operations_json)
        except:
            # Fallback to manual parsing if JSON parse fails
            operations = [
                {"operation": f"Writing: {subtask['description']}", "agent": "writer"},
                {"operation": f"Editing: {subtask['description']}", "agent": "editor"}
            ]
    else:
        # Default operations if parsing fails
        operations = [
            {"operation": f"Writing: {subtask['description']}", "agent": "writer"},
            {"operation": f"Editing: {subtask['description']}", "agent": "editor"}
        ]
    
    return {
        # "current_team": "CONTENT_TEAM",
        "current_agent": operations[0]["agent"],
        "operations": operations
    }

# Specialized agent nodes for research team
def web_search_node(state: HierarchicalAgentState) -> Dict:
    """Web search agent that finds current information online."""
    # Get the current operation and subtask
    operations = state.get("operations", [])
    operation = next((op for op in operations if op["agent"] == "web_search"), 
                    {"operation": "Perform web search"})
    
    subtask = state["current_subtask"]
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()

    web_search_prompt = ChatPromptTemplate.from_messages([
        ("system", WEB_SEARCH_PROMPT + """Use web search tools to find relevant, current information.
        Focus on authoritative sources and recent data.
        Provide a comprehensive but concise summary of your findings."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
     
    agent = create_react_agent(llm, tools, prompt = web_search_prompt)
    
    # Execute the agent
    task_message = f"Task: {operation['operation']}\n\nOriginal context: {state['task']}\n\nUse web search tools to find relevant, current information. Focus on authoritative sources and recent data. Provide a comprehensive but concise summary of your findings."
    
    response = agent.invoke({
        "messages": [HumanMessage(content=task_message)]
    })
    
    return {
        "web_search_result": response["messages"][-1].content,
        "current_agent": "web_search"
    }


def knowledge_base_node(state: HierarchicalAgentState) -> Dict:
    """Knowledge base agent that provides background information."""
    # Get the current operation and subtask
    operations = state.get("operations", [])
    operation = next((op for op in operations if op["agent"] == "knowledge_base"), 
                    {"operation": "Search knowledge base"})
    
    subtask = state["current_subtask"]
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    knowledge_base_prompt = ChatPromptTemplate.from_messages([
        ("system", KNOWLEDGE_BASE_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Partial the prompt with tools and tool names
    knowledge_base_prompt = knowledge_base_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(llm, tools, prompt = knowledge_base_prompt)
    
    # Execute the agent
    task_message = f"Task: {operation['operation']}\n\nOriginal context: {state['task']}\n\nUse your comprehensive knowledge to provide:\n- Core concepts and background information\n- Established theories and principles\n- Historical context or development\n\nFocus on accuracy and depth rather than recency."
    
    response = agent.invoke({
        "messages": [HumanMessage(content=task_message)]
    })
    
    return {
        "knowledge_base_result": response["messages"][-1].content,
        "current_agent": "knowledge_base"
    }

# Specialized agent nodes for analysis team
def data_analyst_node(state: HierarchicalAgentState) -> Dict:
    """Data analyst agent that processes and interprets information."""
    # Get the current operation and subtask
    operations = state.get("operations", [])
    operation = next((op for op in operations if op["agent"] == "data_analyst"), 
                    {"operation": "Analyze data"})
    
    subtask = state["current_subtask"]
    
    # Get research results if available
    research_results = ""
    for key, value in state.get("results", {}).items():
        if "research" in key.lower():
            research_results += f"\n{key}: {value}\n"
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    data_analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", DATA_ANALYST_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Partial the prompt with tools and tool names
    data_analyst_prompt = data_analyst_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(llm, tools, prompt = data_analyst_prompt)
    
    # Execute the agent
    task_message = f"Task: {operation['operation']}\n\nOriginal context: {state['task']}\n\nAvailable research: {research_results}\n\nAnalyze the information provided to:\n- Identify patterns, trends, or correlations\n- Extract key metrics or data points\n- Provide quantitative insights when possible\n- Interpret what the data suggests"
    
    response = agent.invoke({
        "messages": [HumanMessage(content=task_message)]
    })
    
    return {
        "data_analyst_result": response["messages"][-1].content,
        "current_agent": "data_analyst"
    }
    

def critical_thinker_node(state: HierarchicalAgentState) -> Dict:
    """Critical thinking agent that evaluates information logically."""
    # Get the current operation and subtask
    operations = state.get("operations", [])
    operation = next((op for op in operations if op["agent"] == "critical_thinker"), 
                    {"operation": "Critically evaluate"})
    
    subtask = state["current_subtask"]
    
    # Get research and analysis results if available
    previous_results = ""
    for key, value in state.get("results", {}).items():
        previous_results += f"\n{key}: {value}\n"
    
    # Add data analyst result if available
    if "data_analyst_result" in state:
        previous_results += f"\nData Analysis: {state['data_analyst_result']}\n"
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    critical_thinker_prompt = ChatPromptTemplate.from_messages([
        ("system", CRITICAL_THINKER_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Partial the prompt with tools and tool names
    critical_thinker_prompt = critical_thinker_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(llm, tools, prompt = critical_thinker_prompt)
    
    # Execute the agent
    task_message = f"Task: {operation['operation']}\n\nOriginal context: {state['task']}\n\nAvailable information: {previous_results}\n\nCritically evaluate the information by:\n- Identifying assumptions and potential biases\n- Assessing the strength of arguments and evidence\n- Highlighting limitations or gaps\n- Considering alternative interpretations\n- Providing a balanced evaluation"
    
    response = agent.invoke({
        "messages": [HumanMessage(content=task_message)]
    })
    
    return {
        "critical_thinker_result": response["messages"][-1].content,
        "current_agent": "critical_thinker"
    }

# Specialized agent nodes for content team
def writer_node(state: HierarchicalAgentState) -> Dict:
    """Writer agent that creates content."""
    # Get the current operation and subtask
    operations = state.get("operations", [])
    operation = next((op for op in operations if op["agent"] == "writer"), 
                    {"operation": "Write content"})
    
    subtask = state["current_subtask"]
    
    # Get all previous results if available
    previous_results = ""
    for key, value in state.get("results", {}).items():
        previous_results += f"\n{key}: {value}\n"
    
    # Add research team results if available
    if "web_search_result" in state:
        previous_results += f"\nWeb Search: {state['web_search_result']}\n"
    if "knowledge_base_result" in state:
        previous_results += f"\nKnowledge Base: {state['knowledge_base_result']}\n"
    
    # Add analysis team results if available
    if "data_analyst_result" in state:
        previous_results += f"\nData Analysis: {state['data_analyst_result']}\n"
    if "critical_thinker_result" in state:
        previous_results += f"\nCritical Evaluation: {state['critical_thinker_result']}\n"
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_SPECIALIZED_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Partial the prompt with tools and tool names
    writer_prompt = writer_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(llm, tools, prompt = writer_prompt)
    
    # Execute the agent
    task_message = f"Task: {operation['operation']}\n\nOriginal context: {state['task']}\n\nAvailable information: {previous_results}\n\nCreate well-written content that:\n- Directly addresses the original task\n- Incorporates key information from previous research and analysis\n- Is structured logically with clear flow between sections\n- Uses accessible language while maintaining accuracy\n- Is comprehensive but concise"
    
    response = agent.invoke({
        "messages": [HumanMessage(content=task_message)]
    })
    
    
    return {
        "writer_result": response["messages"][-1].content,
        "current_agent": "writer"
    }

def editor_node(state: HierarchicalAgentState) -> Dict:
    """Editor agent that refines and improves content."""
    # Get the current operation and subtask
    operations = state.get("operations", [])
    operation = next((op for op in operations if op["agent"] == "editor"), 
                    {"operation": "Edit content"})
    
    subtask = state["current_subtask"]
    
    # Get writer result if available
    writer_content = state.get("writer_result", "No content available")
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    editor_prompt = ChatPromptTemplate.from_messages([
        ("system", EDITOR_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Partial the prompt with tools and tool names
    editor_prompt = editor_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )
    
    agent = create_react_agent(llm, tools, prompt = editor_prompt)
    
    # Execute the agent
    task_message = f"Task: {operation['operation']}\n\nOriginal context: {state['task']}\n\nContent to edit: {writer_content}\n\nEdit and refine the content to:\n- Improve clarity, conciseness, and flow\n- Ensure logical structure and coherence\n- Fix any grammatical or stylistic issues\n- Verify that it fully addresses the original task\n- Enhance overall quality and readability\n\nProvide the edited version of the content, not just feedback."
    
    response = agent.invoke({
        "messages": [HumanMessage(content=task_message)]
    })
    
    # Combine results from the team
    team_result = response["messages"][-1].content
    
    # Update the results with the completed subtask
    results = state.get("results", {})
    results[state["current_subtask"]["description"]] = team_result
    
    return {
        "results": results,
        "messages": [AIMessage(content=team_result, name="content_team")],
        "current_agent": "editor"
    }

# Combined results function for research team
def combine_research_results(state: HierarchicalAgentState) -> Dict:
    """Combine results from the research team agents."""
    # Get the results from both research agents
    web_search_result = state.get("web_search_result", "No web search results")
    knowledge_base_result = state.get("knowledge_base_result", "No knowledge base results")
    
    # Create a prompt to combine the results
    llm = get_model()
    combine_prompt = PromptTemplate(
        template="""You are tasked with synthesizing research from multiple sources into a cohesive research report.
        
        Web Search Results:
        {web_search}
        
        Knowledge Base Results:
        {knowledge_base}
        
        Create a comprehensive research summary that:
        1. Integrates information from both sources
        2. Emphasizes complementary findings
        3. Resolves any contradictions
        4. Provides a balanced view of current and established knowledge
        
        Your summary should be well-organized, factual, and directly relevant to the original task.
        """,
        input_variables=["web_search", "knowledge_base"]
    )
    
    # Generate the combined result
    response = llm.invoke(
        combine_prompt.format(
            web_search=web_search_result,
            knowledge_base=knowledge_base_result
        )
    )
    
    # Update the results with the completed subtask
    results = state.get("results", {})
    results[state["current_subtask"]["description"]] = response.content
    
    return {
        "results": results,
        "messages": [AIMessage(content=response.content, name="research_team")],
        "current_agent": "research_team"
    }

# Combined results function for analysis team
def combine_analysis_results(state: HierarchicalAgentState) -> Dict:
    """Combine results from the analysis team agents."""
    # Get the results from both analysis agents
    data_analyst_result = state.get("data_analyst_result", "No data analysis results")
    critical_thinker_result = state.get("critical_thinker_result", "No critical thinking results")
    
    # Create a prompt to combine the results
    llm = get_model()
    combine_prompt = PromptTemplate(
        template="""You are tasked with synthesizing analysis from multiple perspectives into a cohesive analytical report.
        
        Data Analysis Results:
        {data_analysis}
        
        Critical Thinking Evaluation:
        {critical_thinking}
        
        Create a comprehensive analytical summary that:
        1. Integrates quantitative insights with qualitative evaluation
        2. Balances factual analysis with critical perspective
        3. Provides a nuanced understanding of the topic
        4. Draws meaningful conclusions based on both analyses
        
        Your summary should be insightful, balanced, and directly relevant to the original task.
        """,
        input_variables=["data_analysis", "critical_thinking"]
    )
    
    # Generate the combined result
    response = llm.invoke(
        combine_prompt.format(
            data_analysis=data_analyst_result,
            critical_thinking=critical_thinker_result
        )
    )
    
    # Update the results with the completed subtask
    results = state.get("results", {})
    results[state["current_subtask"]["description"]] = response.content
    
    return {
        "results": results,
        "messages": [AIMessage(content=response.content, name="analysis_team")],
        "current_agent": "analysis_team"
    }

# Routing functions
def route_to_team(state: HierarchicalAgentState) -> Literal["supervisor", "research_lead", "analysis_lead", "content_lead"]:
    """Route to the appropriate team lead based on the current team."""
    if state.get("is_complete", False):
        return "supervisor"
    
    current_team = state.get("current_team", "")
    
    if current_team == "RESEARCH_TEAM":
        return "research_lead"
    elif current_team == "ANALYSIS_TEAM":
        return "analysis_lead"
    elif current_team == "CONTENT_TEAM":
        return "content_lead"
    else:
        return "supervisor"  # Default

def route_research_agent(state: HierarchicalAgentState) -> Literal["web_search", "knowledge_base", "combine_research"]:
    """Route to the appropriate research agent based on the current agent."""
    current_agent = state.get("current_agent", "")
    
    if current_agent == "web_search":
        return "web_search"
    elif current_agent == "knowledge_base":
        return "knowledge_base"
    else:
        return "combine_research"  # Default to combining results

def route_analysis_agent(state: HierarchicalAgentState) -> Literal["data_analyst", "critical_thinker", "combine_analysis"]:
    """Route to the appropriate analysis agent based on the current agent."""
    current_agent = state.get("current_agent", "")
    
    if current_agent == "data_analyst":
        return "data_analyst"
    elif current_agent == "critical_thinker":
        return "critical_thinker"
    else:
        return "combine_analysis"  # Default to combining results

def route_content_agent(state: HierarchicalAgentState) -> Literal["writer", "editor"]:
    """Route to the appropriate content agent based on the current agent."""
    current_agent = state.get("current_agent", "")
    
    if current_agent == "writer":
        return "writer"
    else:
        return "editor"  # Default to editor

def create_hierarchical_agent_workflow():
    """Create and return the hierarchical agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(HierarchicalAgentState)
    
    # Add nodes for the supervisor and team leads
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_lead", research_lead_node)
    workflow.add_node("analysis_lead", analysis_lead_node)
    workflow.add_node("content_lead", content_lead_node)
    
    # Add nodes for research team
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("knowledge_base", knowledge_base_node)
    workflow.add_node("combine_research", combine_research_results)
    
    # Add nodes for analysis team
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("critical_thinker", critical_thinker_node)
    workflow.add_node("combine_analysis", combine_analysis_results)
    
    # Add nodes for content team
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    
    # Add conditional edges for team routing
    workflow.add_conditional_edges(
        "supervisor",
        route_to_team,
        {
            "supervisor": "supervisor",  # Loop back for next subtask or completion
            "research_lead": "research_lead",
            "analysis_lead": "analysis_lead",
            "content_lead": "content_lead"
        }
    )
    
    # Add conditional edges for research team
    workflow.add_conditional_edges(
        "research_lead",
        route_research_agent,
        {
            "web_search": "web_search",
            "knowledge_base": "knowledge_base",
            "combine_research": "combine_research"
        }
    )
    
    # Add conditional edges for analysis team
    workflow.add_conditional_edges(
        "analysis_lead",
        route_analysis_agent,
        {
            "data_analyst": "data_analyst",
            "critical_thinker": "critical_thinker",
            "combine_analysis": "combine_analysis"
        }
    )
    
    # Add conditional edges for content team
    workflow.add_conditional_edges(
        "content_lead",
        route_content_agent,
        {
            "writer": "writer",
            "editor": "editor"
        }
    )
    
    # Add edges back to the supervisor
    workflow.add_edge("web_search", "knowledge_base")
    workflow.add_edge("knowledge_base", "combine_research")
    workflow.add_edge("combine_research", "supervisor")
    
    workflow.add_edge("data_analyst", "critical_thinker")
    workflow.add_edge("critical_thinker", "combine_analysis")
    workflow.add_edge("combine_analysis", "supervisor")
    
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", "supervisor")

    # Add final conditional edge for completion
    def is_complete(state: HierarchicalAgentState) -> str:
        if state.get("is_complete", False):
            return END
        else:
            return "supervisor"
        
    # Use add_conditional_edges for the final transition
    workflow.add_conditional_edges(
        "supervisor",
        is_complete,
        {
            END: END,
            "supervisor": "supervisor"
        }
    )
    
    # Update run_hierarchical_example to not use thread_config
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())


# Example usage
def run_hierarchical_example(query: str):
    """Run an example query through the hierarchical agent workflow."""
    workflow = create_hierarchical_agent_workflow()
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    thread_config = {"configurable": {"thread_id": "1"}}
    
    # Run the workflow without thread_config
    result = workflow.invoke(initial_state, thread_config)
    
    # Get the final response
    final_response = result["messages"][-1].content
    
    # Get the team results if available
    team_results = {}
    for key, value in result.get("results", {}).items():
        if isinstance(key, str) and isinstance(value, str):
            team_results[key] = value
    
    return {
        "final_response": final_response,
        "team_results": team_results
    }
    
if __name__ == "__main__":
    # Example query
    query = "Explain the key considerations and potential approaches for implementing a sustainable smart city transportation system."
    
    # Run the example
    result = run_hierarchical_example(query)
    
    print("\n--- Hierarchical Agent Workflow Results ---")
    print("\nTeam Results:")
    for subtask, result_text in result["team_results"].items():
        print(f"\nSubtask: {subtask}")
        print(f"Result excerpt: {result_text[:150]}...")
    
    print("\nFinal Response:")
    print(result["final_response"])