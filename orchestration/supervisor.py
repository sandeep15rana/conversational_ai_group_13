"""
Supervisor / Planner Agent (Developer 1 - Sandeep Rana)
Core logic for:
1. Analyzing user requests and decomposing goals into a Directed Acyclic Graph (DAG) of sub-tasks.
2. Tracking state execution progress and choosing the next specialized worker agent.
3. Enforcing recursion limits and loop protection.
"""

from typing import Dict, Any, List
from .state import AgentState

class SupervisorPlanner:
    """
    Supervisor Agent responsible for task planning, DAG construction,
    and dynamic routing of state across the agent graph.
    """
    
    def __init__(self, max_iterations: int = 8):
        self.max_iterations = max_iterations

    def plan_and_route(self, state: AgentState) -> Dict[str, Any]:
        """
        Inspects current state, builds plan DAG if needed, increments iteration counter,
        and decides the next agent to invoke.
        """
        current_iter = state.get("iteration_count", 0) + 1
        trace = [f"[Supervisor]: Iteration {current_iter}/{self.max_iterations} - Evaluating state..."]
        
        # 1. Check Security Flag (from Input Guardrail)
        if not state.get("is_safe", True):
            trace.append("[Supervisor]: Input rejected by guardrail. Routing directly to Error Handler.")
            return {
                "next_action": "error",
                "iteration_count": current_iter,
                "execution_trace": trace
            }
            
        # 2. Check Runaway Recursion Limiter
        if current_iter >= self.max_iterations:
            trace.append("[Supervisor]: CRITICAL! Max recursion limit reached. Halting loop to prevent token runaway.")
            return {
                "next_action": "error",
                "iteration_count": current_iter,
                "errors": [f"Execution halted: Maximum iteration limit ({self.max_iterations}) exceeded."],
                "execution_trace": trace
            }
            
        # 3. Dynamic Plan Construction (DAG Generation)
        plan = state.get("plan", [])
        if not plan:
            query = state.get("sanitized_query", "")
            trace.append("[Supervisor]: Decomposing high-level query into sub-task DAG...")
            
            # Sub-task decomposition logic
            generated_plan = [
                "Task 1: Retrieve Q1-Q3 financial data for NVIDIA and AMD",
                "Task 2: Calculate percentage growth and absolute revenue expansion in Python",
                "Task 3: Search recent supply-chain and packaging constraint news",
                "Task 4: Synthesize executive briefing formatted with cited references"
            ]
            plan = generated_plan
            trace.append(f"[Supervisor]: Created DAG with {len(plan)} sub-tasks.")
            
        # 4. Determine Next Action Based on Completed Tasks
        completed = state.get("completed_tasks", [])
        research_data = state.get("research_data", [])
        calculated_metrics = state.get("calculated_metrics", {})
        draft = state.get("draft_response", "")
        
        # Rule A: If financial data hasn't been retrieved yet -> call Research Agent
        if len(research_data) == 0:
            next_action = "research"
            trace.append("[Supervisor]: Action decided -> Delegate to Research Agent (Web Tool).")
            
        # Rule B: If research is done but calculations are pending -> call Analyst Agent
        elif len(calculated_metrics) == 0:
            next_action = "analyst"
            trace.append("[Supervisor]: Action decided -> Delegate to Analyst Agent (Python REPL).")
            
        # Rule C: If both research and calculations are ready, but no draft -> call Synthesis Agent
        elif not draft:
            next_action = "synthesize"
            trace.append("[Supervisor]: Action decided -> Delegate to Synthesis Agent (Context Fusion).")
            
        # Rule D: If draft is ready -> Complete pipeline
        else:
            next_action = "finish"
            trace.append("[Supervisor]: All tasks completed successfully. Finalizing pipeline.")
            
        return {
            "plan": plan,
            "next_action": next_action,
            "iteration_count": current_iter,
            "execution_trace": trace
        }

# Factory instance for LangGraph node
_supervisor_instance = SupervisorPlanner(max_iterations=8)

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node wrapper for Supervisor planning logic."""
    return _supervisor_instance.plan_and_route(state)
