"""
Enhanced Human-in-the-Loop Agent

A more robust implementation of a human-in-the-loop workflow where the agent can request human input
during execution and incorporate human feedback using advanced LangGraph interrupts.
"""

import asyncio
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional, Literal, Union
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command

from utils import get_model, get_common_tools


"""
Enhanced Human-in-the-Loop Agent with Improved Workflow

A robust implementation focusing on a dedicated Human-in-the-Loop node 
with advanced routing and feedback mechanisms.
"""

import asyncio
from typing import Dict, List, Any, Annotated, Sequence, TypedDict, Optional, Literal
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command

from utils import get_model, get_common_tools


# Define the state for the human-in-the-loop workflow
class HumanInTheLoopState(TypedDict):
    """State for human-in-the-loop workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    human_input_needed: bool
    human_input_reason: str
    human_input_response: Optional[str]
    internal_thoughts: List[str]
    confidence: float
    feedback_on_plan: Optional[str]
    next_node: Optional[str]  # New field to track routing


async def initialize_state(state: HumanInTheLoopState) -> Dict:
    """Initialize the state for the workflow."""
    return {
        "human_input_needed": False,
        "human_input_reason": "",
        "human_input_response": None,
        "internal_thoughts": [],
        "confidence": 1.0,
        "feedback_on_plan": None,
        "next_node": None  # Initialize next_node
    }


async def task_planner(state: HumanInTheLoopState) -> Dict:
    """Plan the approach to solving the task and prepare for human feedback."""
    query = state["messages"][-1].content
    
    # Create the planner
    llm = get_model()
    tools = get_common_tools()
    
    # Define a specific prompt for task planning
    task_planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced task planning assistant using web search and other 
        available tools to create a comprehensive and well-researched plan.

        Your goal is to:
        1. Analyze the user's request thoroughly
        2. Use available tools to gather relevant information
        3. Create a detailed, step-by-step plan
        4. Identify potential challenges and areas needing human input
        5. Provide rationale for each step of the plan

        When creating the plan, consider:
        - Breaking down complex tasks into manageable steps
        - Using tool-gathered information to inform your planning
        - Highlighting areas that may require human expertise or input
        """),
       ("placeholder", "{messages}"),
    ])
    
    # Partial the prompt with tools and tool names
    task_planning_prompt = task_planning_prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )

    # Create the React agent
    agent = create_react_agent(
        llm, 
        tools, 
        prompt=task_planning_prompt
    )
    
    response = agent.invoke({
        "messages": query
    })
    
    plan = response["messages"][-1].content
    
    print(f"Generated Plan:\n{plan}")
    # planner_prompt = PromptTemplate(
    #     template="""You are a task planning assistant. Your job is to analyze the user's request
    #     and create a step-by-step plan for addressing it.
        
    #     User request: {query}
        
    #     Create a detailed plan with 3-5 steps for completing this task. For each step:
    #     1. Describe what needs to be done
    #     2. Identify potential challenges or areas needing human input
    #     3. Explain the rationale behind each step
        
    #     Format your response as a structured, clear plan.
    #     """,
    #     input_variables=["query"]
    # )
    
    
    # # Generate the plan
    # chain = planner_prompt | llm
    # response = chain.invoke({"query": query})
    # plan = response.content
    
    return {
        "internal_thoughts": [f"Initial Task Plan: {plan}"],
        "human_input_needed": True,  # Trigger human input node
        "human_input_reason": f"Please review the following task plan:\n\n{plan}\n\nDo you approve this plan?",
        "next_node": None  # Will be set by human feedback node
    }


async def human_feedback_node(state: HumanInTheLoopState) -> Dict:
    """
    Dedicated node for handling human feedback and routing.
    
    This node:
    1. Presents the plan or request for human review
    2. Captures human feedback
    3. Determines the next routing based on feedback
    """
    llm = get_model()
    
    # Use interrupt to get human input
    feedback_request = state.get("human_input_reason", "No specific reason provided.")
    
    # Request human input with routing options
    human_response = interrupt(f"""
    {feedback_request}

    Possible actions:
    1. Approve and continue to next step
    2. Request modifications
    3. Cancel or restart

    Please respond with:
    - '1' to approve and continue
    - '2' followed by your suggestions for modification
    - '3' to cancel or restart
    """)
    
    # Process human feedback
    if isinstance(human_response, str):
        response_type = human_response[0]
        
        if response_type == '1':
            # Approved, route to confidence assessment
            return {
                "human_input_needed": False,
                "human_input_response": "Plan approved",
                "feedback_on_plan": "Approved by human",
                "next_node": "assess_confidence",
                "internal_thoughts": state["internal_thoughts"] + ["Human approved the plan"]
            }
        
        elif response_type == '2':
            # Modifications requested
            modification_details = human_response[1:].strip()
            
            # Use LLM to revise plan based on human feedback
            revision_prompt = PromptTemplate(
                template="""You are a task planning assistant. Revise the original plan 
                based on the following human feedback:
                
                Original Plan: {original_plan}
                
                Human Feedback: {human_feedback}
                
                Provide a revised plan addressing the feedback while maintaining 
                the core objectives of the original plan.
                """,
                input_variables=["original_plan", "human_feedback"]
            )
            
            original_plan = state["internal_thoughts"][-1].replace("Initial Task Plan: ", "")
            
            # Generate revised plan
            revision_chain = revision_prompt | llm
            revised_response = revision_chain.invoke({
                "original_plan": original_plan,
                "human_feedback": modification_details
            })
            
            return {
                "human_input_needed": True,
                "human_input_reason": f"Revised Plan:\n\n{revised_response.content}\n\nDo you approve this revised plan?",
                "internal_thoughts": state["internal_thoughts"] + [
                    f"Human Modification Request: {modification_details}",
                    f"Revised Task Plan: {revised_response.content}"
                ],
                "next_node": None  # Will trigger another human feedback iteration
            }
        
        elif response_type == '3':
            # Cancel or restart
            return {
                "human_input_needed": False,
                "human_input_response": "Process cancelled by human",
                "feedback_on_plan": "Cancelled",
                "next_node": END,  # Or another appropriate node
                "internal_thoughts": state["internal_thoughts"] + ["Process cancelled by human"]
            }
        
    # Fallback for unexpected input
    return {
        "human_input_needed": False,
        "human_input_response": "Unexpected input",
        "next_node": END,
        "internal_thoughts": state["internal_thoughts"] + ["Unexpected human input received"]
    }


async def assess_confidence(state: HumanInTheLoopState) -> Dict:
    """Assess confidence level and determine if additional human input is needed."""
    query = state["messages"][-1].content
    llm = get_model()
    
    confidence_prompt = PromptTemplate(
        template="""Assess the confidence in executing the following task plan:
        
        Task: {query}
        
        Current Plan: {task_plan}
        
        Rate your confidence from 0.0 to 1.0. If confidence is below 0.7:
        1. Explain specific areas of uncertainty
        2. Recommend what additional information or input would help
        
        Output format:
        Confidence: [0.0-1.0]
        Uncertainty Areas: [list of specific concerns]
        Additional Input Needed: [detailed recommendations]
        """,
        input_variables=["query", "task_plan"]
    )
    
    # Extract the most recent task plan from internal thoughts
    task_plans = [
        thought.replace("Initial Task Plan: ", "").replace("Revised Task Plan: ", "") 
        for thought in state["internal_thoughts"] 
        if "Task Plan:" in thought
    ]
    task_plan = task_plans[-1] if task_plans else "No detailed plan available"
    
    # Generate confidence assessment
    chain = confidence_prompt | llm
    response = chain.invoke({
        "query": query,
        "task_plan": task_plan
    })
    
    # Parse the response
    confidence = 1.0
    uncertainty_areas = ""
    additional_input = ""
    
    for line in response.content.split("\n"):
        if line.startswith("Confidence:"):
            try:
                confidence = float(line.split(":")[1].strip())
            except:
                confidence = 1.0
        elif line.startswith("Uncertainty Areas:"):
            uncertainty_areas = line.split(":")[1].strip()
        elif line.startswith("Additional Input Needed:"):
            additional_input = line.split(":")[1].strip()
    
    # Determine next steps based on confidence
    if confidence < 0.7:
        return {
            "confidence": confidence,
            "human_input_needed": True,
            "human_input_reason": f"""
            Confidence is low for the current task plan.
            
            Uncertainty Areas: {uncertainty_areas}
            
            Additional Input Needed: {additional_input}
            
            Please provide more context or guidance.
            """,
            "internal_thoughts": state["internal_thoughts"] + [
                f"Confidence Assessment: {response.content}",
                f"Confidence Level: {confidence}"
            ],
            "next_node": None  # Will trigger human feedback node
        }
    else:
        return {
            "confidence": confidence,
            "human_input_needed": False,
            "next_node": "generate_response",
            "internal_thoughts": state["internal_thoughts"] + [
                f"Confidence Assessment: {response.content}",
                f"Confidence Level: {confidence}"
            ]
        }


async def generate_response(state: HumanInTheLoopState) -> Dict:
    """Generate a response based on the task and any human input."""
    query = state["messages"][-1].content
    
    # Create the agent
    llm = get_model()
    tools = get_common_tools()
    
    PROMPT = """You are an assistant that works collaboratively with humans.
    Incorporate any human feedback or input received during the planning process.
    
    Provide a comprehensive and thoughtful response that addresses the original query
    while reflecting on any guidance or modifications suggested during the workflow.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
    ])
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt=prompt)
    
    # Execute the agent
    response = await agent.ainvoke({
        "input": query,
        "chat_history": state.get("messages", [])
    })
    
    # Final human review of the response
    review_request = f"""
    I've prepared a response to your request. Please review:

    {response["messages"][-1].content}

    Actions:
    1. Accept the response
    2. Request modifications
    3. Restart the process

    Respond with:
    - '1' to accept
    - '2' followed by modification requests
    - '3' to restart
    """
    
    review_response = interrupt(review_request)
    
    # Process review response
    if isinstance(review_response, str):
        response_type = review_response[0]
        
        if response_type == '1':
            # Response accepted
            return {
                "messages": [response["messages"][-1]],
                "next_node": END,
                "internal_thoughts": state.get("internal_thoughts", []) + ["Final response accepted by human"]
            }
        
        elif response_type == '2':
            # Modifications requested
            modification_details = review_response[1:].strip()
            
            # Revise response
            revision_prompt = PromptTemplate(
                template="""Revise the response based on human feedback:
                
                Original Query: {query}
                Original Response: {original_response}
                
                Human Modification Request: {modification_request}
                
                Provide a modified response addressing the specific feedback.
                """,
                input_variables=["query", "original_response", "modification_request"]
            )
            
            revision_chain = revision_prompt | llm
            revised_response = revision_chain.invoke({
                "query": query,
                "original_response": response["messages"][-1].content,
                "modification_request": modification_details
            })
            
            return {
                "messages": [AIMessage(content=revised_response.content)],
                "next_node": END,
                "internal_thoughts": state.get("internal_thoughts", []) + [
                    "Response modification requested",
                    f"Modification details: {modification_details}",
                    f"Revised response: {revised_response.content}"
                ]
            }
        
        elif response_type == '3':
            # Restart process
            return {
                "messages": [],
                "next_node": START,
                "internal_thoughts": state.get("internal_thoughts", []) + ["Process restarted at user's request"]
            }
    
    # Fallback
    return {
        "messages": [response["messages"][-1]],
        "next_node": END,
        "internal_thoughts": state.get("internal_thoughts", []) + ["Response processed with default handling"]
    }


def route_next_node(state: HumanInTheLoopState) -> str:
    """
    Route to the next node based on the state's next_node attribute.
    Provides a flexible routing mechanism.
    """
    next_node = state.get("next_node")
    
    # Default routing logic
    if next_node:
        return next_node
    elif state.get("human_input_needed"):
        return "human_feedback"
    else:
        return "generate_response"


def create_human_in_the_loop_workflow():
    """Create and return the human-in-the-loop workflow."""
    # Create the workflow graph
    workflow = StateGraph(HumanInTheLoopState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("task_planner", task_planner)
    workflow.add_node("human_feedback", human_feedback_node)
    workflow.add_node("assess_confidence", assess_confidence)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "task_planner")
    
    # Add conditional edges with more flexible routing
    workflow.add_conditional_edges(
        "task_planner",
        route_next_node,
        {
            "human_feedback": "human_feedback",
            "assess_confidence": "assess_confidence",
            "generate_response": "generate_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "human_feedback",
        route_next_node,
        {
            "task_planner": "task_planner",
            "assess_confidence": "assess_confidence",
            "generate_response": "generate_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
            "assess_confidence",
            route_next_node,
            {
                "human_feedback": "human_feedback",
                "generate_response": "generate_response",
                END: END
            }
        )
    
    workflow.add_conditional_edges(
        "generate_response",
        route_next_node,
        {
            END: END
        }
    )
    
    # Set the default end point
    workflow.add_edge("generate_response", END)
    
    # Compile the workflow
    return workflow.compile(checkpointer=MemorySaver())


async def run_human_in_the_loop_example(query: str, simulate_human_input: Dict[str, str] = None):
    """
    Run an example query through the human-in-the-loop workflow.
    
    Args:
        query: The user query to process
        simulate_human_input: Dictionary mapping interrupt points to simulated responses
    """
    # Create a custom interrupt handler
    from langchain_core.callbacks import CallbackManager
    from langchain_core.callbacks.base import BaseCallbackHandler

    class SimpleInterruptHandler(BaseCallbackHandler):
        def __init__(self, simulate_inputs=None):
            self.simulate_inputs = simulate_inputs or {}
            self.interrupt_count = 0
        
        async def on_interrupt(self, message, run_id, **kwargs):
            """Handle interrupt by simulating user input or requesting from console."""
            self.interrupt_count += 1
            print(f"\nInterrupt #{self.interrupt_count} triggered: {message}")
            
            if self.simulate_inputs:
                # Priority-based simulation
                simulation_keys = [
                    f'interrupt_{self.interrupt_count}', 
                    'default_response'
                ]
                
                for key in simulation_keys:
                    if key in self.simulate_inputs:
                        response = self.simulate_inputs[key]
                        print(f"Simulated response for {key}: {response}")
                        return response
            
            # Fallback to console input
            print(f"\nHuman input requested: {message}")
            user_input = input("Your response: ")
            return user_input
    
    # Initialize the workflow
    workflow = create_human_in_the_loop_workflow()
    
    # Initialize the state with the query
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Create the interrupt handler with simulated inputs
    interrupt_handler = SimpleInterruptHandler(simulate_inputs=simulate_human_input)
    
    # Set up the config with our interrupt handler
    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [interrupt_handler]
    }
    
    print(f"\n=== Processing query: {query} ===")
    
    try:
        # Execute the workflow
        result = await workflow.ainvoke(initial_state, config)
        
        # Extract final results
        final_response = "No response generated."
        if result.get("messages"):
            # Get the last AI message
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_response = msg.content
                    break
                    
        internal_thoughts = result.get("internal_thoughts", [])
        confidence = result.get("confidence", 0.0)
        human_input_needed = result.get("human_input_needed", False)
        human_input_reason = result.get("human_input_reason", "")
        human_input_response = result.get("human_input_response", "")
        
        return {
            "final_response": final_response,
            "internal_thoughts": internal_thoughts,
            "confidence": confidence,
            "human_input_needed": human_input_needed,
            "human_input_reason": human_input_reason,
            "human_input_response": human_input_response
        }
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        return {
            "final_response": f"Error: {str(e)}",
            "internal_thoughts": [],
            "confidence": 0.0,
            "human_input_needed": False,
            "human_input_reason": "",
            "human_input_response": ""
        }


if __name__ == "__main__":
    import asyncio
    
    # Example queries with simulated inputs
    queries = [
        {
            "query": "Create a comprehensive workout plan for improving overall fitness.",
            "simulated_input": {
                "interrupt_1": "2Add more emphasis on cardiovascular exercises",
                "interrupt_2": "1",  # Accept the response
                "default_response": "1"  # Fallback for other interrupts
            }
        },
        {
            "query": "Help me plan a trip to Japan for two weeks.",
            "simulated_input": {
                "interrupt_1": "2Include more cultural experiences and local cuisine recommendations",
                "interrupt_2": "1",  # Accept the response
                "default_response": "1"  # Fallback for other interrupts
            }
        }
    ]
    
    # Run examples
    async def run_examples():
        print("\n=== Human-in-the-Loop Agent Results ===")
        
        for i, example in enumerate(queries):
            print(f"\n--- Example {i+1} ---")
            print(f"Query: {example['query']}")
            
            result = await run_human_in_the_loop_example(
                example["query"], 
                simulate_human_input=example["simulated_input"]
            )
            
            print(f"\nConfidence: {result['confidence']}")
            print(f"Human Input Needed: {result['human_input_needed']}")
            if result['human_input_needed']:
                print(f"Human Input Reason: {result['human_input_reason']}")
                print(f"Human Input Response: {result['human_input_response']}")
            
            print("\nFinal Response:")
            print(result["final_response"])
            
            print("\nInternal Thoughts (for debugging):")
            for thought in result["internal_thoughts"]:
                print(f"- {thought}")
    
    # Run the examples
    asyncio.run(run_examples())