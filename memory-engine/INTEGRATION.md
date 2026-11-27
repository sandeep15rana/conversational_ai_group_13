# Memory Engine – Integration Guide

> **Module:** Dual-Layer Memory & Context Engine (Developer 3 – Rajarshi)
> **For:** Sandeep (orchestration), Pooja (worker agents), Harshavardhan (guardrails & evaluation), Hemlata (report)
> **Status:** Every code block in this guide was run against a copy of the real repo code before it was written here. Nobody's files were changed.

---

## 1. What this module does (30-second version)

| Layer | What it remembers | Where | How long |
|---|---|---|---|
| **Short-term** | Last 20 turns of the current session, plus scratchpad notes | RAM (per user + session) | Until the session is idle for 1 hour |
| **Long-term** | User preferences, facts about the user, conclusions of past questions | ChromaDB on disk (per user) | Until deleted |

It plugs into the LangGraph pipeline at **two points**:

```
START → input_guardrail → [memory_recall] → supervisor → workers … → output_guardrail → [memory_save] → END
```

- **`memory_recall`**: before planning, finds what we already know about this user.
- **`memory_save`**: after the answer is final, saves anything new worth remembering.

---

## 2. One-time setup (beginner steps)

> **Use Python 3.12.** ChromaDB and its model runtime are tested on 3.12. Python 3.14 support is not confirmed.

**Step 1. Create a virtual environment** (once, from the repo root):

```powershell
py -3.12 -m venv .venv
```
```powershell
.venv\Scripts\activate
```

**Step 2. Install the memory engine** into the same environment you run the graph from:

```powershell
pip install -e ./memory-engine
```
```powershell
pip install langgraph
```

`-e` means "editable": if Rajarshi updates the code, you get the change without reinstalling.

**Step 3. First run downloads the embedding model** (~80 MB, only once):
- Model: `all-MiniLM-L6-v2` (sentence-transformers), ONNX build.
- Source: ChromaDB's own server, **not HuggingFace**, which works on the company network.
- Saved to: `C:\Users\<you>\.cache\chroma\onnx_models\all-MiniLM-L6-v2\`.
- No PyTorch needed.

**Step 4. Check it works:**

```powershell
python -c "from memory_engine import memory_recall_node; print('memory engine OK')"
```

---

## 3. The contract (what memory reads and writes)

Memory only uses fields that **already exist** in `orchestration/state.py`. **No change to `AgentState` is needed.**

| Node | Reads from state | Writes to state |
|---|---|---|
| `memory_recall_node` | `user_id`, `session_id`, `sanitized_query`, `is_safe` | `recalled_context`, `user_preferences`, `execution_trace` |
| `memory_save_node` | `user_id`, `session_id`, `final_output`, `is_safe`, `is_hallucination_detected` | `execution_trace` |

**Meaning of `recalled_context`:**

| Value | Meaning |
|---|---|
| A text block | Memory found something relevant |
| `""` (empty string) | Memory ran, but nothing relevant is known yet (e.g. a new user) |
| `None` | Memory was skipped (unsafe input, no `user_id`) or failed |

**Example `recalled_context`:**
```
User preferences:
- User prefers format: table
- User prefers include citations: true
Known facts about the user:
- User is a portfolio manager
- User works at Acme Corp
Relevant past conclusions:
- Past question: Compare NVIDIA and AMD data center revenue growth from Q1 to Q3. | Conclusion: NVIDIA's data center revenue rose ... 36.28% ...
Recent conversation (this session):
- User: and what about Intel?
```

- **Preferences and facts** are always included (the user's "profile").
- **Past conclusions** are included only when they are relevant to the query.

**Example `user_preferences`:**
```python
{"format": "table", "include_citations": True, "detail_level": "concise"}
```

Possible keys: `format` (`bullet_points` / `table` / `json` / `paragraphs`), `detail_level` (`concise` / `detailed`), `include_citations` (bool), `include_charts` (bool).

**Helpers you can import** (all from `memory_engine`):

| Helper | For |
|---|---|
| `memory_recall_node`, `memory_save_node` | The two graph nodes (section 4) |
| `memory_context_for_synthesis(state)` | Pooja's `SynthesisRequest.memory_context` (section 5) |
| `build_supervisor_memory_block(state)` | An LLM supervisor prompt (section 4.4) |
| `save_scratchpad` / `load_scratchpad` | Session working notes for any agent (section 4.5) |
| `no_memory_node` | "Without memory" baseline for evaluation (section 6) |

**Safety guarantees:**
- **Fails soft.** If memory breaks (even on odd input such as `user_id=123`), the trace shows a warning and the pipeline still answers.
- **Per-user isolation.** One user can never see another user's memories.
- **No `user_id` = no memory.** Unknown users are never pooled into one shared bucket.
- **User ids are pseudonymized.** Only a keyed hash (HMAC-SHA256) of the id is stored, not the raw id. The key comes from `MEMORY_ID_HASH_KEY` if set; otherwise a random key is created once in the store folder. For real deployments, set `MEMORY_ID_HASH_KEY` so the key doesn't sit next to the data.
- **PII masking before saving.** Covers emails (including "name at domain dot com"), phone numbers, API keys, JWT/Bearer/Slack tokens, passwords and PINs, SSN, Aadhaar, PAN, passport, IBAN, card numbers, IP addresses and labelled dates of birth.
  - **Numbers alone are never treated as PII**, so revenue figures, years and price lists stay intact. An ambiguous number is masked only if it passes a checksum (cards: Luhn; Aadhaar: Verhoeff) or comes with a label ("phone", "mobile", "call / text / reach me on", "my number is", "aadhaar", "pin is"). A bare "number" is not a label, so "the total number was 9500000000" stays.
  - Unambiguous formats (`+91 98765 43210`, `555-123-4567`, `020 7946 0958`) are always masked.
  - It is pattern-based: unusual formats can slip through, so don't treat it as a full DLP system.
- **Prompt-injection hardening.** Common attack phrasings ("ignore your instructions", "reveal the system prompt", "you are now...") are not saved. If one appears in a question, the conclusion is kept with the question replaced by `[question omitted]`. Preferences are always stored in a fixed form ("User prefers format: table"), never as the user's raw sentence.
  - This is pattern-based, so paraphrased attacks can slip through.
  - Harshavardhan's input guardrail is the main defence; the supervisor block also says the current request wins over memory.
- **Unsafe input** (`is_safe=False`): nothing is recalled and nothing is saved.
- **Hallucinated answer** (`is_hallucination_detected=True`): the answer is **not** memorized and is not replayed as recent conversation.

---

## 4. Sandeep – `orchestration/graph.py`

### 4.1 The change: 4 edits, nothing else

```diff
 from .state import AgentState
 from .supervisor import supervisor_node
+from memory_engine import memory_recall_node, memory_save_node  # Dev 3: Memory Engine
 from .mock_agents import (
     mock_input_guardrail,
-    mock_memory_recall,
     mock_research_agent,
     mock_analyst_agent,
     mock_synthesis_agent,
@@
     # 2. Add Nodes
     workflow.add_node("input_guardrail", mock_input_guardrail)
-    workflow.add_node("memory_recall", mock_memory_recall)
+    workflow.add_node("memory_recall", memory_recall_node)    # Dev 3: real memory recall
+    workflow.add_node("memory_save", memory_save_node)        # Dev 3: save new memories
     workflow.add_node("supervisor", supervisor_node)
@@
     # 6. Terminal Edges
-    workflow.add_edge("output_guardrail", END)
+    workflow.add_edge("output_guardrail", "memory_save")      # Dev 3: save after the answer is final
+    workflow.add_edge("memory_save", END)
     workflow.add_edge("error_handler", END)
```

**No changes needed in:** `state.py`, `supervisor.py`, `mock_agents.py` (`mock_memory_recall` can stay there unused), `test_orchestrator.py`, `run_orchestration_demo.py`.

### 4.2 Full file after integration (copy-paste ready)

Replace the whole of `orchestration/graph.py` with this:

```python
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
from memory_engine import memory_recall_node, memory_save_node  # Dev 3: Memory Engine
from .mock_agents import (
    mock_input_guardrail,
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
    workflow.add_node("memory_recall", memory_recall_node)    # Dev 3: real memory recall
    workflow.add_node("memory_save", memory_save_node)        # Dev 3: save new memories
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
    workflow.add_edge("output_guardrail", "memory_save")      # Dev 3: save after the answer is final
    workflow.add_edge("memory_save", END)
    workflow.add_edge("error_handler", END)
    
    # 7. Compile with Checkpointer for State Persistence
    if checkpointer is None:
        checkpointer = MemorySaver()
        
    return workflow.compile(checkpointer=checkpointer)
```

### 4.3 Run and verify

From the repo root:

```powershell
python -m unittest orchestration.test_orchestrator -v
```
Expected: `Ran 4 tests ... OK`. All 4 of your tests pass unchanged.

```powershell
python run_orchestration_demo.py
```

Expected new lines in the trace:
```
03. [Memory Engine]: Recalling long-term memory for user 'sandeep_rana'...
04. [Memory Engine]: Recalled 0 relevant memories, 0 profile facts and 0 preferences [].
...
23. [Memory Engine]: Consolidating memories from this turn...
24. [Memory Engine]: Saved via 'rules' - added 1, updated 0, duplicates 0, rejected 0, evicted 0.
```

Run the demo a **second time**: line 04 becomes `Recalled 1 relevant memories` and line 24 says `duplicates 1`, meaning memory persisted and the same fact was not stored twice.

**Tip:** while testing, keep test data out of the real store:
```powershell
$env:MEMORY_PERSIST_DIR = "$env:TEMP\memory_test_store"
```

### 4.4 Optional: when the supervisor becomes LLM-based

Your supervisor is rule-based today, so it needs nothing. When you move to an LLM prompt, add one line:

```python
from memory_engine import build_supervisor_memory_block

prompt = SUPERVISOR_PROMPT.format(query=state["sanitized_query"]) + "\n\n" + build_supervisor_memory_block(state)
```

`build_supervisor_memory_block(state)` returns `""` when there is no memory, so it is always safe to append.

### 4.5 Optional: scratchpad for intermediate results

Any node can keep working notes for the rest of the session (a plan, partial findings), without adding fields to `AgentState`:

```python
from memory_engine import save_scratchpad, load_scratchpad

save_scratchpad(state, "plan", plan)                       # inside the supervisor
previous_plan = load_scratchpad(state, "plan", default=[])  # in a later turn of the same session
```

- Scoped to `user_id` + `session_id`, and expires with the session (1 hour idle).
- Never raises: if memory is unavailable, it returns `False` / the default.

---

## 5. Pooja – Synthesis Agent

**No change to any of your files.** Your `SynthesisRequest` already has `memory_context: list[str]`.

Wherever the graph builds a `SynthesisRequest` (when your agent replaces `mock_synthesis_agent`), pass memory like this:

```python
from worker_agents.schemas import SynthesisRequest
from worker_agents.synthesis_agent import SynthesisAgent
from memory_engine import memory_context_for_synthesis

request = SynthesisRequest(
    original_query=state["sanitized_query"],
    research_findings=...,      # as today
    analytical_findings=...,    # as today
    memory_context=memory_context_for_synthesis(state),   # Dev 3: recalled memories, one per line
)
response = SynthesisAgent(model=llm).run(request)
```

`memory_context_for_synthesis(state)` turns `recalled_context` into a clean list:
```python
["User prefers format: table (said: ...)", "User is a portfolio manager", "Past question: ... | Conclusion: ..."]
```

Your system prompt already says "use only the supplied ... memory context", so formatting preferences flow straight into the answer. `state["user_preferences"]` (a dict) is also available if you want to branch on, for example, `format == "table"`.

**Verified:** your real `SynthesisAgent` was run with a mock LLM. The memory lines arrived in `request.memory_context` and in the prompt sent to the model.

---

## 6. Harshavardhan – Guardrails & Evaluation

**No code change needed.** How memory works with your parts:

| Your output | What memory does |
|---|---|
| `sanitized_query` | Memory reads this, never `raw_query` |
| `is_safe = False` | Recall skipped, nothing saved. Attack text never reaches memory |
| `is_hallucination_detected = True` | The answer is **not** saved as a conclusion (preferences and facts from the user still are) |

PII is also masked again inside memory before saving (defence in depth).

**For your evaluation**, every turn writes fixed-format trace lines you can parse:
```
[Memory Engine]: Recalled <n> relevant memories, <f> profile facts and <m> preferences [...]
[Memory Engine]: Saved via '<rules|llm>' - added a, updated u, duplicates d, rejected r, evicted e.
[Memory Engine]: WARNING - ...        (memory failed, pipeline continued)
```

**Ablation "with vs without memory":** build the graph twice. For the "without" run, use the baseline node:

```python
from memory_engine import no_memory_node

workflow.add_node("memory_recall", no_memory_node)   # baseline: recalls nothing
workflow.add_edge("output_guardrail", END)           # and no memory_save
```

Do **not** use `mock_memory_recall` as the baseline: it injects fake preferences, so it isn't "without memory".

**Retrieval quality (already measured, real model, 20 labelled queries):**

| Metric | Value |
|---|---|
| Hit@1 | 0.80 |
| Hit@3 | 1.00 |
| MRR | 0.90 |

Reproduce:
```powershell
memory-engine\.venv\Scripts\python memory-engine\eval\retrieval_eval.py
```

---

## 7. Hemlata – for the report

| You need | Where |
|---|---|
| Memory architecture | Section 1 of this guide, plus `README.md` |
| Cosine similarity formula | `cos(A,B) = (A·B) / (‖A‖‖B‖)`. Chroma stores distance = 1 − cos |
| Ranking formula | `score = cos + 0.1 × 0.5^(age_days / 30)`, then MMR: `λ·score − (1−λ)·max_cos(selected)`, λ = 0.7 |
| Memory flowchart steps | Section 8 below |
| "With vs without memory" | Run `memory-engine\.venv\Scripts\python memory-engine\demo_memory.py`. The memory part is real; the final answer text comes from a small stand-in formatter, because the full LLM Synthesis Agent isn't wired into the graph yet |
| Retrieval metrics | Table in section 6 |

---

## 8. How it works inside (algorithm)

**Recall (read path):**
1. **Profile:** load the user's preferences and newest 10 facts (always, whatever the query).
2. **Embed** the query with all-MiniLM-L6-v2 → 384-number vector.
3. **Search** that user's memories by cosine similarity. The search is **exact**: each user has at most 500 memories, so all of them are scored. This avoids approximate-search misses on a large shared collection.
4. **Filter:** drop anything below 0.35 similarity.
5. **Merge** chunks that belong to the same long memory.
6. **Rank:** similarity + small recency bonus (half-life 30 days).
7. **Diversify** with MMR so you don't get five near-copies; keep the top 5.
8. **Budget:** cut the text to 600 tokens, add the last 4 turns of the session.

**Save (write path):**
1. Take only the **new** turns of this session. Each turn has a sequence number, so nothing is skipped or processed twice.
2. **Extract:** preferences and facts from **user** messages; one conclusion from the final answer.
   - Rules by default. If `OPENAI_API_KEY` is set, an LLM is used, with rules as fallback.
   - One-off requests ("I need the revenue table for Q1") are not stored as preferences.
3. Drop candidates with confidence below 0.6.
4. **Mask PII** and **reject prompt injection**.
5. **Exact duplicate** (same normalized text) → refresh it, don't add.
6. **Same preference key** (e.g. `format`) → the newer value replaces the old. Different keys are never merged.
7. **Near duplicate** (cosine ≥ 0.90, facts and conclusions) → update in place.
8. **Chunk** long text: ≤ 200 tokens, 1 sentence overlap (the model reads at most 256 tokens). Handles Chinese and Hindi punctuation, and text without spaces such as long URLs.
9. **Cap:** at most 500 memories per user; the lowest-importance, oldest go first.

---

## 9. Settings (environment variables or `.env`)

All are optional; the defaults are tested.

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_PERSIST_DIR` | `memory-engine/.memory_store` (editable install) or `~/.memory_engine/store` (normal install) | Where ChromaDB saves files (`none` = RAM only) |
| `MEMORY_TOP_K` | 5 | Memories recalled per query |
| `MEMORY_MIN_SIMILARITY` | 0.35 | Minimum cosine similarity to count as relevant |
| `MEMORY_DUPLICATE_THRESHOLD` | 0.90 | At or above this, a memory counts as a duplicate |
| `MEMORY_MAX_MEMORIES_PER_USER` | 500 | Per-user cap |
| `MEMORY_CONTEXT_TOKEN_BUDGET` | 600 | Maximum size of `recalled_context` |
| `MEMORY_MAX_PROFILE_FACTS` | 10 | Newest facts always shown, whatever the query |
| `MEMORY_SHORT_TERM_MAX_TURNS` | 20 | Turns kept per session |
| `MEMORY_SHORT_TERM_TTL_SECONDS` | 3600 | Idle session expiry |
| `MEMORY_EMBEDDING_MODEL_DIR` | (Chroma cache) | Folder with an offline copy of the model |
| `MEMORY_ID_HASH_KEY` | key file in the store folder | Secret for hashing user ids. Changing it makes existing memories unreachable |
| `OPENAI_API_KEY` | not set | If set, the LLM extractor is used (`MEMORY_LLM_MODEL`, default `gpt-4o-mini`) |

Bad values fail at start-up with a clear message (e.g. `MEMORY_MIN_SIMILARITY=1.5` is rejected).

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'memory_engine'` | Run `pip install -e ./memory-engine` in the **same** venv you run the graph from |
| Model download blocked | On any machine with internet, download `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz` and extract it so you have `<folder>\onnx\model.onnx`. Then set `MEMORY_EMBEDDING_MODEL_DIR=<folder>` |
| Trace shows `[Memory Engine]: WARNING - recall failed` | Memory is down but the pipeline still answered. The warning shows the reason (usually the model or the disk path) |
| `Collection ... was built with '<model>'` | The store was created with a different embedding model. Use a new `MEMORY_PERSIST_DIR` or delete `.memory_store` |
| Tests leave data in the real store | Set `MEMORY_PERSIST_DIR` to a temp folder (see 4.3) |
| Need to wipe one user | `from memory_engine import get_default_engine; get_default_engine().forget_user("user_id")`. Deleted memories are never returned again; ChromaDB reclaims the disk space during its own compaction |
| Need to wipe everything | Delete the `memory-engine/.memory_store` folder |
| `[Memory Engine]: No user_id in state - memory skipped` | The state had no `user_id`. Memory only works for identified users |
| Embedding model failed once and every request now fails fast | After a failed load, the model is retried at most once a minute (so requests don't each wait on the network). Fix the cause, then wait a minute or restart |

**Known limits**

- **One process per store.** Short-term memory and the write lock live in process memory. Run the graph in one worker process per memory store (fine for this project). Several worker processes sharing one store would need a shared cache (e.g. Redis) and a server-mode ChromaDB.
- **Session limit.** At most 1,000 sessions are held in RAM; the least recently used are dropped first.

---

## 11. How this integration was validated

| Check | Result |
|---|---|
| Memory engine unit + validation tests (dedupe, isolation, PII, injection, cap, chunking, TTL, fail-soft) | 229 pass, 95% line coverage |
| Independent code review (separate agent, three rounds) | Final verdict "ready to commit"; all findings fixed or documented; each has a regression test in `tests/test_review_regressions.py` |
| Real-model tests (MiniLM + ChromaDB, semantic recall, chunk sizes) | Pass |
| Sandeep's 4 unit tests on a **copy** of the repo with the edits in section 4 | `Ran 4 tests ... OK` |
| Sandeep's `run_orchestration_demo.py` on that copy, run twice | Memory recalled on the 2nd run; duplicate detected, not re-stored |
| Pooja's real `SynthesisAgent` receiving `memory_context` | Pass |
| Prompt-injection query | Recall skipped, nothing stored |
| Original files of Sandeep / Pooja / others | **Unchanged** (`git status` shows changes only inside `memory-engine/`) |

`memory-engine/tests/test_orchestration_integration.py` re-applies the section 4 edits to Sandeep's **current** `graph.py` on every test run. If `graph.py` changes so the edits no longer apply, that test fails and tells us to update this guide.

```powershell
cd memory-engine
```
```powershell
.venv\Scripts\python -m pytest -q
```
