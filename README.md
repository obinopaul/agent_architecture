# Multi-Agent Architectures with LangChain and LangGraph + other Agentic structures

This repository contains implementations of various multi-agent architectures and LangGraph features.


<p align="center">
  <img src="Agentic Architecture.jpeg" width="500" alt="Image description">
</p>


## Basic Architectures

### 1. Parallel
Multiple agents work simultaneously on tasks, with designated input and output points.

### 2. Sequential
Agents work in order, passing results from one to the next, with clear input and output points.

### 3. Loop
Similar to sequential but includes a feedback mechanism where output can be cycled back as input.

### 4. Router
One agent directs input to different output paths based on task requirements.

### 5. Aggregator
Multiple input streams are combined by an agent into a single output.

### 6. Network
A mesh-like configuration where multiple interconnected agents collaborate, with designated input and output points.

### 7. Hierarchical
A structured tree-like organization where a main agent delegates to subordinate agents.

### 11. Human-in-the-Loop
Workflow that can request human input during execution for tasks requiring human judgment.

### 12. Agent as a Tool
Architecture where a ReAct agent can be called as a tool by another ReAct agent.

## Getting Started

### Prerequisites

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
```

### Running Examples

To run all basic architectures with sample queries:

```bash
python main.py
```

To run a specific architecture:

```bash
python main.py --architecture parallel
```

With a custom query:

```bash
python main.py --architecture router --query "What are the best practices for microservice architecture?"
```

## File Structure

### Basic Architectures
- `utils.py` - Shared utilities and base classes
- `aggregator_agents.py` - Aggregator Agents
- `parallel_agents.py` - Parallel architecture implementation
  - `parallel_agents_main.py` - Parallel Architecture Implementation 2
- `sequential_agents.py` - Sequential architecture implementation
- `loop_agents.py` - Loop architecture implementation
- `router_agents.py` - Router architecture implementation
- `router_agents_with_command.py` - Router architecture implementation with Command
- `aggregator_agents.py` - Aggregator architecture implementation
- `network_agents.py` - Network architecture implementation
- `hierarchical_agents.py` - Hierarchical architecture implementation
- `react_agent_as_a_tool.py` - Agent as a Tool
- `human_in_the_loop.py` - Human in the Loop
- `main.py` - Main script to run basic architecture examples

#   a g e n t _ a r c h i t e c t u r e 
 
 