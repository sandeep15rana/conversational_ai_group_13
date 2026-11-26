"""
Master Shared State Schema (AgentState)
Defines the typed dictionary contract used across all nodes in the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer helper to merge dictionaries across state updates."""
    merged = a.copy()
    merged.update(b)
    return merged

class AgentState(TypedDict):
    # Session & Identity
    session_id: str
    user_id: str
    
    # Layer 1: Input Guardrail & Sanitization (Dev 4)
    raw_query: str
    is_safe: bool
    sanitized_query: str
    security_flag: Optional[str]
    
    # Layer 2: Memory & Context (Dev 3)
    recalled_context: Optional[str]
    user_preferences: Dict[str, Any]
    
    # Layer 3: Supervisor & Planning (Dev 1 - Sandeep Rana)
    plan: List[str]                  # Execution DAG sub-tasks
    completed_tasks: List[str]       # Tasks successfully completed
    current_step: int                # Index in plan
    next_action: str                 # "research", "analyst", "synthesize", "error", "finish"
    iteration_count: int             # Recursion guard counter
    max_iterations: int              # Safety ceiling (e.g., 8)
    
    # Layer 4: Worker Agent Observations (Dev 2)
    # Annotated with operator.add or custom merger to accumulate findings
    research_data: Annotated[List[Dict[str, Any]], operator.add]
    calculated_metrics: Annotated[Dict[str, Any], merge_dicts]
    
    # Layer 5: Synthesis & Output Guardrail (Dev 2 & Dev 4)
    draft_response: str
    final_output: str
    is_hallucination_detected: bool
    
    # Error Handling & Telemetry Trace
    errors: Annotated[List[str], operator.add]
    execution_trace: Annotated[List[str], operator.add]
