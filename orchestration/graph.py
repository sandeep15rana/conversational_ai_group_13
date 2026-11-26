"""
Master LangGraph StateGraph Architecture (Developer 1 - Sandeep Rana)
Connects all nodes, defines conditional edge routing, implements state persistence,
and attaches error recovery fallbacks.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .supervisor import supervisor_node
from .mock_agents import (
    mock_input_guardrail,
    mock_memory_recall,
    mock_research_agent,
    mock_analyst_agent,
    mock_synthesis_agent,
    mock_output_guardrail
)

# ==============================================================================
# Error Recovery Node
# ==============================================================================
def error_recovery_node(state: AgentState) -> Dict[str, Any]:
    """
    Fallback recovery node:
    Provides graceful degradation when guardrails trip, tools error, or recursion limits are hit.
    """
    errors = state.get("errors", ["Unknown execution anomaly."])
    trace = ["[Error Handler]: Triggered fallback recovery handler."]
    
    fallback_message = (
        "### System Execution Notice\n"
        "The automated multi-agent workflow encountered a safety or operational constraint:\n\n"
    )
    for err in errors:
        fallback_message += f"- [!] {err}\n"
        
    fallback_message += (
        "\nThe request could not be safely completed under current security and execution policies. "
        "Please refine your query or contact the system administrator."
    )
    
    trace.append("[Error Handler]: Safe fallback message generated.")
    return {
        "final_output": fallback_message,
        "execution_trace": trace
    }

# ==============================================================================
# Conditional Edge Router
# ==============================================================================
def route_supervisor_decision(state: AgentState) -> Literal["research_agent", "analyst_agent", "synthesis_agent", "error_handler", "output_guardrail"]:
    """
    Inspects Supervisor's decision and routes to the appropriate node.
    """
    action = state.get("next_action", "error")
    
    if action == "research":
        return "research_agent"
    elif action == "analyst":
        return "analyst_agent"
    elif action == "synthesize":
        return "synthesis_agent"
    elif action == "finish":
        return "output_guardrail"
    else:
        return "error_handler"

# ==============================================================================
# Master Graph Construction Function
# ==============================================================================
def build_orchestrator_graph(checkpointer: Any = None):
    """
    Constructs and compiles the complete LangGraph StateGraph workflow.
    
    Graph Topology:
    START ──▶ input_guardrail ──▶ memory_recall ──▶ supervisor
                                                        │
                      ┌─────────────────────────────────┼─────────────────────────┐
                      ▼                                 ▼                         ▼
             [ research_agent ]                 [ analyst_agent ]         [ synthesis_agent ]
                      │                                 │                         │
                      └───────────────┬─────────────────┘                         ▼
                                      ▼                                  [ output_guardrail ]
                                (supervisor)                                      │
                                      │ (if error / recursion limit)              ▼
                                      ▼                                         [ END ]
                               [ error_handler ] ──▶ [ END ]
    """
    # 1. Initialize StateGraph with our shared AgentState schema
    workflow = StateGraph(AgentState)
    
    # 2. Add Nodes
    workflow.add_node("input_guardrail", mock_input_guardrail)
    workflow.add_node("memory_recall", mock_memory_recall)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", mock_research_agent)
    workflow.add_node("analyst_agent", mock_analyst_agent)
    workflow.add_node("synthesis_agent", mock_synthesis_agent)
    workflow.add_node("output_guardrail", mock_output_guardrail)
    workflow.add_node("error_handler", error_recovery_node)
    
    # 3. Define Linear Ingestion Edges
    workflow.add_edge(START, "input_guardrail")
    workflow.add_edge("input_guardrail", "memory_recall")
    workflow.add_edge("memory_recall", "supervisor")
    
    # 4. Conditional Edge: Supervisor routing
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {
            "research_agent": "research_agent",
            "analyst_agent": "analyst_agent",
            "synthesis_agent": "synthesis_agent",
            "output_guardrail": "output_guardrail",
            "error_handler": "error_handler"
        }
    )
    
    # 5. Worker Loops: return back to supervisor to track DAG progress
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analyst_agent", "supervisor")
    workflow.add_edge("synthesis_agent", "supervisor")
    
    # 6. Terminal Edges
    workflow.add_edge("output_guardrail", END)
    workflow.add_edge("error_handler", END)
    
    # 7. Compile with Checkpointer for State Persistence
    if checkpointer is None:
        checkpointer = MemorySaver()
        
    return workflow.compile(checkpointer=checkpointer)
