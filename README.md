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

## Advanced Architectures

### 13. Isolated Environment Agent
A secure code execution agent that uses Daytona sandboxes to run code in completely isolated environments. This architecture provides:

- **Secure Code Execution**: All code runs in isolated Daytona sandboxes, preventing any risk to the host system
- **Multi-language Support**: Execute Python code, shell commands, and other scripts safely
- **File Management**: Upload, download, and manage files within sandboxes
- **Git Integration**: Clone repositories directly into sandboxes for development workflows
- **Resource Management**: Automatic sandbox creation, management, and cleanup

**Use Cases:**
- Running untrusted or experimental code safely
- Data analysis and visualization in isolated environments
- Testing and development workflows
- Educational coding environments
- AI-generated code execution with security guarantees

### 14. Filesystem Planner Agent
An advanced planning agent that creates and manages structured todo.md files while executing complex tasks. This architecture provides:

- **Structured Planning**: Creates comprehensive todo.md files with task breakdowns
- **Dynamic Plan Updates**: Updates plans in real-time as tasks are completed
- **Dual Operation Modes**: Works with actual filesystem files or in-memory state management
- **Progress Tracking**: Maintains detailed history of task execution and completion
- **Flexible Workspace Management**: Organizes work in structured directories

**Use Cases:**
- Project planning and execution with documentation
- Complex multi-step task automation
- Development workflow management
- Educational project structuring
- Collaborative planning with persistent documentation

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
TAVILY_API_KEY=your_tavily_api_key
```

For the Isolated Environment Agent, you'll also need to set up Daytona:

1. Install Daytona CLI:
   ```bash
   # macOS/Linux
   brew install daytonaio/cli/daytona
   
   # Windows
   # Download from https://github.com/daytonaio/daytona/releases
   ```

2. Authenticate with Daytona:
   ```bash
   daytona login
   ```

3. (Optional) Initialize Daytona MCP server for enhanced AI integration:
   ```bash
   daytona mcp init
   ```

### Running Examples

To run all architectures (including advanced ones) with sample queries:

```bash
python main.py
```

To run a specific basic architecture:

```bash
python main.py --architecture parallel
```

To run the advanced architectures:

```bash
# Isolated Environment Agent
python main.py --architecture isolated_environment

# Filesystem Planner Agent
python main.py --architecture filesystem_planner
```

With custom queries:

```bash
# Basic architecture example
python main.py --architecture router --query "What are the best practices for microservice architecture?"

# Advanced architecture examples
python main.py --architecture isolated_environment --query "Create a data analysis script that processes CSV files and generates plots"

python main.py --architecture filesystem_planner --query "Plan and build a REST API with authentication and database integration"
```

### Advanced Architecture Usage

#### Isolated Environment Agent Examples

```bash
# Secure code execution
python main.py --architecture isolated_environment --query "Execute this Python code safely: import pandas as pd; df = pd.read_csv('data.csv'); print(df.head())"

# Data analysis with visualization
python main.py --architecture isolated_environment --query "Create a machine learning model to predict house prices and save the results"

# Git repository analysis
python main.py --architecture isolated_environment --query "Clone a Python repository and run its test suite in isolation"
```

#### Filesystem Planner Agent Examples

```bash
# Project planning and execution
python main.py --architecture filesystem_planner --query "Create a complete web scraping project with documentation and testing"

# Development workflow
python main.py --architecture filesystem_planner --query "Plan and implement a microservice with Docker containerization"

# Educational project structure
python main.py --architecture filesystem_planner --query "Set up a data science project with proper directory structure and documentation"
```

## File Structure

### Core Files
- `utils.py` - Shared utilities, base classes, and system prompts
- `main.py` - Main script to run all architecture examples
- `requirements.txt` - Python dependencies for all architectures

### Basic Architectures
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

### Advanced Architectures
- `isolated_environment_agent.py` - Isolated Environment Agent with Daytona integration
- `daytona_tools.py` - Custom tools for Daytona sandbox operations
- `filesystem_planner_agent.py` - Filesystem Planner Agent with todo.md management

#### Advanced Architecture Details

**Isolated Environment Agent (`isolated_environment_agent.py`)**
- Secure code execution using Daytona sandboxes
- Custom workflow with sandbox initialization, code execution, and cleanup
- Integration with `daytona_tools.py` for comprehensive sandbox management

**Daytona Tools (`daytona_tools.py`)**
- `DaytonaSandboxManager` - Core sandbox management class
- `CreateSandboxTool` - Create new isolated sandboxes
- `DestroySandboxTool` - Clean up sandbox resources
- `ExecuteCommandTool` - Run shell commands in sandboxes
- `CodeExecutionTool` - Execute Python code safely
- `FileUploadTool` / `FileDownloadTool` - File management within sandboxes
- `ListSandboxesTool` - Monitor available sandboxes
- `GitCloneTool` - Clone repositories into sandboxes

**Filesystem Planner Agent (`filesystem_planner_agent.py`)**
- Structured planning with todo.md file management
- Dual operation modes: filesystem and state-based
- Custom workflow with workspace initialization, plan creation, task execution, and finalization

**Filesystem Tools**
- `FileSystemTool` - Basic filesystem operations (create_dir, read_file, write_file, etc.)
- `TodoManagerTool` - Todo.md creation, updating, and task management
- `PlanUpdateTool` - Plan refinement and reorganization capabilities

# agent_architecture





