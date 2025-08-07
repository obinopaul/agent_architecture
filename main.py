"""
Multi-Agent Architecture Demonstration

This script showcases various multi-agent architectures implemented with LangChain and LangGraph.
"""

import asyncio
from typing import Dict, Any
import argparse

from langchain_core.messages import HumanMessage

# Import all architectures
from parallel_agents import create_parallel_agent_workflow, run_parallel_example
from sequential_agents import create_sequential_agent_workflow, run_sequential_example
from loop_agents import create_loop_agent_workflow, run_loop_example
from router_agents import create_router_agent_workflow, run_router_example
from aggregator_agents import create_aggregator_agent_workflow, run_aggregator_example
from network_agents import create_network_agent_workflow, run_network_example
from hierarchical_agents import create_hierarchical_agent_workflow, run_hierarchical_example
from isolated_environment_agent import create_isolated_environment_workflow, run_isolated_environment_example
from filesystem_planner_agent import create_filesystem_planner_workflow, run_filesystem_planner_example

# Sample queries for each architecture
SAMPLE_QUERIES = {
    "parallel": "What are the potential environmental and economic impacts of large-scale renewable energy adoption?",
    "sequential": "Compare and contrast different approaches to artificial intelligence ethics frameworks.",
    "loop": "Write a summary of the current state of quantum computing research for a general audience.",
    "router": "Create a personal financial planning guide for recent college graduates.",
    "aggregator": "Analyze the future of work considering automation, remote work trends, and changing skill requirements.",
    "network": "Explain how blockchain technology might be applied to improve supply chain transparency and security.",
    "hierarchical": "Develop a comprehensive strategy for a city to reduce its carbon footprint over the next decade.",
    "isolated_environment": "Create a Python script that analyzes a dataset, generates visualizations, and saves the results to files in a secure sandbox environment.",
    "filesystem_planner": "Plan and implement a complete web scraping project with proper directory structure, documentation, and step-by-step execution tracking."
}

async def run_all_examples():
    """Run examples of all architectures with their sample queries."""
    results = {}
    
    # Parallel architecture (async)
    print("\n=== Running Parallel Architecture ===")
    parallel_result = await run_parallel_example(SAMPLE_QUERIES["parallel"])
    results["parallel"] = parallel_result
    print("✓ Parallel architecture completed")
    
    # Sequential architecture
    print("\n=== Running Sequential Architecture ===")
    sequential_result = run_sequential_example(SAMPLE_QUERIES["sequential"])
    results["sequential"] = sequential_result["final_response"]
    print("✓ Sequential architecture completed")
    
    # Loop architecture
    print("\n=== Running Loop Architecture ===")
    loop_result = run_loop_example(SAMPLE_QUERIES["loop"], max_iterations=2)
    results["loop"] = loop_result["final_message"]
    print("✓ Loop architecture completed")
    
    # Router architecture
    print("\n=== Running Router Architecture ===")
    router_result = run_router_example(SAMPLE_QUERIES["router"])
    results["router"] = router_result["final_message"]
    print("✓ Router architecture completed")
    
    # Aggregator architecture (async)
    print("\n=== Running Aggregator Architecture ===")
    aggregator_result = await run_aggregator_example(SAMPLE_QUERIES["aggregator"])
    results["aggregator"] = aggregator_result["final_response"]
    print("✓ Aggregator architecture completed")
    
    # Network architecture
    print("\n=== Running Network Architecture ===")
    network_result = run_network_example(SAMPLE_QUERIES["network"])
    results["network"] = network_result["final_response"]
    print("✓ Network architecture completed")
    
    # Hierarchical architecture
    print("\n=== Running Hierarchical Architecture ===")
    hierarchical_result = run_hierarchical_example(SAMPLE_QUERIES["hierarchical"])
    results["hierarchical"] = hierarchical_result["final_response"]
    print("✓ Hierarchical architecture completed")
    
    # Isolated Environment architecture
    print("\n=== Running Isolated Environment Architecture ===")
    isolated_env_result = run_isolated_environment_example(SAMPLE_QUERIES["isolated_environment"])
    results["isolated_environment"] = isolated_env_result["final_response"]
    print("✓ Isolated Environment architecture completed")
    
    # Filesystem Planner architecture
    print("\n=== Running Filesystem Planner Architecture ===")
    filesystem_planner_result = run_filesystem_planner_example(SAMPLE_QUERIES["filesystem_planner"])
    results["filesystem_planner"] = filesystem_planner_result["final_response"]
    print("✓ Filesystem Planner architecture completed")
    
    return results

def run_specific_architecture(architecture: str, query: str):
    """Run a specific architecture with a given query."""
    print(f"\n=== Running {architecture.capitalize()} Architecture ===")
    
    if architecture == "parallel":
        result = asyncio.run(run_parallel_example(query))
        return result
    elif architecture == "sequential":
        result = run_sequential_example(query)
        return result["final_response"]
    elif architecture == "loop":
        result = run_loop_example(query, max_iterations=2)
        return result["final_message"]
    elif architecture == "router":
        result = run_router_example(query)
        return result["final_message"]
    elif architecture == "aggregator":
        result = asyncio.run(run_aggregator_example(query))
        return result["final_response"]
    elif architecture == "network":
        result = run_network_example(query)
        return result["final_response"]
    elif architecture == "hierarchical":
        result = run_hierarchical_example(query)
        return result["final_response"]
    else:
        return f"Unknown architecture: {architecture}"

def main():
    """Main function to parse arguments and run examples."""
    parser = argparse.ArgumentParser(description="Demonstrate multi-agent architectures")
    parser.add_argument("--architecture", "-a", type=str, choices=[
        "parallel", "sequential", "loop", "router", "aggregator", "network", "hierarchical", "all"
    ], default="all", help="Which architecture to run (default: all)")
    parser.add_argument("--query", "-q", type=str, help="Custom query to run (if not provided, a sample query is used)")
    
    args = parser.parse_args()
    
    if args.architecture == "all":
        # Run all architectures with their sample queries
        results = asyncio.run(run_all_examples())
        
        # Display a summary of results
        print("\n=== Architecture Results Summary ===")
        for arch, result in results.items():
            print(f"\n{arch.upper()} ARCHITECTURE RESULT (excerpt):")
            excerpt = result[:200] + "..." if len(result) > 200 else result
            print(excerpt)
    else:
        # Run a specific architecture
        query = args.query if args.query else SAMPLE_QUERIES[args.architecture]
        result = run_specific_architecture(args.architecture, query)
        
        # Display the result
        print(f"\n=== {args.architecture.upper()} ARCHITECTURE RESULT ===")
        print(result)

if __name__ == "__main__":
    main()


