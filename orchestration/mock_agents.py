"""
Mock Implementations of Dependent Modules
Allows Developer 1 (Sandeep Rana) to test the Orchestration Engine & State Graph
standalone without waiting for Developer 2, 3, or 4 implementations.
"""

import re
from typing import Dict, Any
from .state import AgentState

# ==============================================================================
# MOCK 1: Input Guardrail & PII Sanitizer (Simulating Developer 4)
# ==============================================================================
def mock_input_guardrail(state: AgentState) -> Dict[str, Any]:
    """
    Simulates input security verification:
    - Scans for prompt injection attacks
    - Masks simulated PII (emails, phone numbers)
    """
    query = state.get("raw_query", "")
    trace = ["[Guardrail 1]: Scanning input for prompt injection and PII..."]
    
    # Simple injection trigger for testing
    injection_patterns = ["ignore previous instructions", "system override", "jailbreak", "dan mode"]
    if any(pat in query.lower() for pat in injection_patterns):
        trace.append("[Guardrail 1]: ALERT! Malicious prompt injection pattern detected.")
        return {
            "is_safe": False,
            "security_flag": "INJECTION_ATTACK_DETECTED",
            "sanitized_query": "",
            "execution_trace": trace,
            "errors": ["Input rejected due to adversarial prompt injection policy."]
        }
        
    # PII Scrubbing
    sanitized = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", query)
    sanitized = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", sanitized)
    
    trace.append("[Guardrail 1]: Passed security scan. Input sanitized.")
    return {
        "is_safe": True,
        "security_flag": None,
        "sanitized_query": sanitized,
        "execution_trace": trace
    }


# ==============================================================================
# MOCK 2: Memory Engine (Simulating Developer 3)
# ==============================================================================
def mock_memory_recall(state: AgentState) -> Dict[str, Any]:
    """
    Simulates ChromaDB long-term memory retrieval:
    Retrieves stored user preferences and historical guidelines.
    """
    user_id = state.get("user_id", "default_user")
    trace = [f"[Memory Engine]: Recalling long-term preferences for user '{user_id}'..."]
    
    # Simulated vector similarity lookup
    recalled_profile = (
        "User Preference: Deliver concise executive briefing with bullet points, "
        "explicit percentage growth calculations, and cited SEC source references."
    )
    prefs = {
        "format": "bulleted_briefing",
        "detail_level": "executive",
        "include_citations": True
    }
    
    trace.append("[Memory Engine]: Retrieved preferences: 'bulleted briefing format with citations'.")
    return {
        "recalled_context": recalled_profile,
        "user_preferences": prefs,
        "execution_trace": trace
    }


# ==============================================================================
# MOCK 3: Research Agent (Simulating Developer 2)
# ==============================================================================
def mock_research_agent(state: AgentState) -> Dict[str, Any]:
    """
    Simulates external search via DuckDuckGo / Yahoo Finance API.
    Fetches raw SEC 10-Q numbers and supply chain news.
    """
    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)
    task_desc = plan[step_idx] if step_idx < len(plan) else "General market research"
    
    trace = [f"[Research Agent]: Executing search tool for task: '{task_desc}'..."]
    
    # Synthetic retrieved data
    findings = {
        "task": task_desc,
        "source": "SEC Form 10-Q / Reuters Intelligence",
        "data": {
            "NVDA_data_center": {"Q1": 22.6, "Q2": 26.3, "Q3": 30.8, "unit": "Billion USD"},
            "AMD_data_center": {"Q1": 2.3, "Q2": 2.8, "Q3": 3.5, "unit": "Billion USD"},
            "supply_chain_risks": "TSMC CoWoS advanced packaging capacity constraints remain primary delivery bottleneck through 2026."
        }
    }
    
    trace.append("[Research Agent]: Found verified revenue numbers and TSMC CoWoS supply-chain report.")
    
    completed = state.get("completed_tasks", []) + [task_desc]
    return {
        "research_data": [findings],
        "completed_tasks": completed,
        "current_step": step_idx + 1,
        "execution_trace": trace
    }


# ==============================================================================
# MOCK 4: Analytical Agent (Simulating Developer 2)
# ==============================================================================
def mock_analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    Simulates sandboxed Python REPL execution with pandas:
    Performs deterministic percentage growth calculations.
    """
    plan = state.get("plan", [])
    step_idx = state.get("current_step", 0)
    task_desc = plan[step_idx] if step_idx < len(plan) else "Calculate growth metrics"
    
    trace = [f"[Analyst Agent]: Executing Python REPL for task: '{task_desc}'..."]
    
    # Deterministic math calculation
    nvda_q1 = 22.6
    nvda_q3 = 30.8
    nvda_growth = ((nvda_q3 - nvda_q1) / nvda_q1) * 100
    
    amd_q1 = 2.3
    amd_q3 = 3.5
    amd_growth = ((amd_q3 - amd_q1) / amd_q1) * 100
    
    calc_results = {
        "NVDA_Q1_to_Q3_growth_pct": round(nvda_growth, 2),
        "AMD_Q1_to_Q3_growth_pct": round(amd_growth, 2),
        "NVDA_absolute_gain_billion": round(nvda_q3 - nvda_q1, 2),
        "AMD_absolute_gain_billion": round(amd_q3 - amd_q1, 2),
        "calculation_verified": True
    }
    
    trace.append(f"[Analyst Agent]: Computed NVDA Growth: {nvda_growth:.2f}%, AMD Growth: {amd_growth:.2f}%.")
    
    completed = state.get("completed_tasks", []) + [task_desc]
    return {
        "calculated_metrics": calc_results,
        "completed_tasks": completed,
        "current_step": step_idx + 1,
        "execution_trace": trace
    }


# ==============================================================================
# MOCK 5: Synthesis Agent (Simulating Developer 2)
# ==============================================================================
def mock_synthesis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Simulates Synthesis / Reviewer agent:
    Combines research findings and calculated math into a formatted briefing.
    """
    trace = ["[Synthesis Agent]: Fusing research findings and mathematical calculations..."]
    
    calcs = state.get("calculated_metrics", {})
    prefs = state.get("user_preferences", {})
    
    nvda_pct = calcs.get("NVDA_Q1_to_Q3_growth_pct", 36.28)
    amd_pct = calcs.get("AMD_Q1_to_Q3_growth_pct", 52.17)
    
    briefing = (
        "### Executive Intelligence Briefing: NVIDIA vs. AMD Data Center Performance\n\n"
        f"**1. Revenue & Growth Trajectory (Q1 to Q3):**\n"
        f"- **NVIDIA:** Grew from $22.6B in Q1 to $30.8B in Q3, registering **+{nvda_pct}% growth** "
        f"(+$8.20B absolute expansion). [Source: SEC 10-Q]\n"
        f"- **AMD:** Grew from $2.3B in Q1 to $3.5B in Q3, registering **+{amd_pct}% growth** "
        f"(+$1.20B absolute expansion). [Source: SEC 10-Q]\n\n"
        "**2. Critical Supply-Chain Bottlenecks:**\n"
        "- The dominant operational risk remains **TSMC CoWoS advanced packaging capacity**.\n"
        "- Sub-tier substrate supply allocations will constrain delivery schedules through 2026.\n\n"
        "**3. Strategic Takeaway:**\n"
        "- While AMD shows higher percentage growth from a smaller base, NVIDIA captures >85% of total incremental revenue."
    )
    
    trace.append("[Synthesis Agent]: Generated executive briefing aligning with user's preferred format.")
    return {
        "draft_response": briefing,
        "execution_trace": trace
    }


# ==============================================================================
# MOCK 6: Output Guardrail & Hallucination Checker (Simulating Developer 4)
# ==============================================================================
def mock_output_guardrail(state: AgentState) -> Dict[str, Any]:
    """
    Simulates output verification:
    Checks that the draft is grounded in research data and math calculations.
    """
    trace = ["[Guardrail 2]: Verifying output grounding and checking for hallucinations..."]
    draft = state.get("draft_response", "")
    
    # Grounding check: verify that calculated numbers appear in the draft
    calcs = state.get("calculated_metrics", {})
    nvda_pct = str(calcs.get("NVDA_Q1_to_Q3_growth_pct", "36.28"))
    
    if nvda_pct in draft:
        trace.append("[Guardrail 2]: Output passed! Verified factual grounding against Python calculations.")
        return {
            "final_output": draft,
            "is_hallucination_detected": False,
            "execution_trace": trace
        }
    else:
        trace.append("[Guardrail 2]: WARNING: Numerical hallucination suspected.")
        return {
            "final_output": draft,
            "is_hallucination_detected": True,
            "execution_trace": trace,
            "errors": ["Output guardrail flagged potential inconsistency with raw calculations."]
        }
