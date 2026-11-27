"""
================================================================================
DEVELOPER 3: DUAL-LAYER MEMORY & CONTEXT ENGINE (RAJARSHI) - DEMO
AIMLCZG521: Conversational AI - Group 13
================================================================================
Shows "Response WITH memory vs WITHOUT memory" using the real stack:
ChromaDB + all-MiniLM-L6-v2 + rule-based extraction.

  Session 1: the user introduces themselves, states preferences, asks a question.
  Session 2: a new session asks a related question in different words.

The final "answer" is rendered by a small stand-in formatter that follows the
recalled preferences (in the full system Pooja's Synthesis Agent does this).

Run:  .venv\\Scripts\\python demo_memory.py
"""

from __future__ import annotations

import sys
import tempfile
import textwrap

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from memory_engine import (
    MemoryConfig,
    MemoryEngine,
    build_supervisor_memory_block,
    memory_recall_node,
    memory_save_node,
    set_default_engine,
)

FINDINGS = {
    "NVIDIA": ("$22.6B", "$30.8B", "+36.28%"),
    "AMD": ("$2.3B", "$3.5B", "+52.17%"),
}
SOURCE = "SEC Form 10-Q"


def line(char: str = "=", width: int = 80) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    line()
    print(f" >>> {title}")
    line()


def render_answer(preferences: dict) -> str:
    """Stand-in for the Synthesis Agent: same facts, formatted the way the user prefers."""
    cite = preferences.get("include_citations") is True
    if preferences.get("format") == "table":
        rows = ["| Company | Q1 | Q3 | Growth |", "|---|---|---|---|"]
        rows += [f"| {c} | {q1} | {q3} | {g} |" for c, (q1, q3, g) in FINDINGS.items()]
        body = "\n".join(rows)
    elif preferences.get("format") == "bullet_points":
        body = "\n".join(f"- {c}: {q1} -> {q3} ({g})" for c, (q1, q3, g) in FINDINGS.items())
    else:
        body = (
            "NVIDIA's data center revenue rose from $22.6B in Q1 to $30.8B in Q3, a gain of 36.28%, "
            "while AMD grew from $2.3B to $3.5B, a gain of 52.17%. AMD grew faster in percentage terms "
            "from a much smaller base, but NVIDIA captured most of the absolute growth."
        )
        if preferences.get("detail_level") != "concise":
            body += " Supply constraints at TSMC remain the main risk to both companies' delivery schedules."
    if cite:
        body += f"\n\nSource: {SOURCE}"
    return body


def state(query: str, session_id: str, final_output: str = "") -> dict:
    return {
        "user_id": "demo_user", "session_id": session_id, "sanitized_query": query,
        "is_safe": True, "final_output": final_output, "is_hallucination_detected": False,
    }


def main() -> None:
    line()
    print("   BITS PILANI - WILP (AIMLCZG521) | CONVERSATIONAL AI - GROUP 13")
    print("   DEVELOPER 3: DUAL-LAYER MEMORY & CONTEXT ENGINE - Owner: Rajarshi")
    line()

    store_dir = tempfile.mkdtemp(prefix="memory_demo_")
    engine = MemoryEngine(config=MemoryConfig.from_env(persist_dir=store_dir, collection_name="demo_memory"))
    set_default_engine(engine)
    print(f"[i] Embedding model : {engine.embedder.name} ({engine.embedder.dimension} dims)")
    print(f"[i] Extractor       : {engine.extractor.name}")
    print(f"[i] Vector store    : ChromaDB at {store_dir}")

    # ------------------------------------------------------------------ session 1
    section("SESSION 1 - user introduces themselves")
    q1 = (
        "Hi, I'm a portfolio manager and I work at Acme Corp. "
        "I prefer tables with citations. Compare NVIDIA and AMD data center revenue growth from Q1 to Q3."
    )
    print(f"[User] {q1}\n")
    s1 = state(q1, "session_001")
    recall1 = memory_recall_node(s1)
    print(f"[Memory] recalled_context = {recall1['recalled_context']!r}  (first visit, nothing known)")
    answer1 = render_answer({})
    save1 = memory_save_node({**s1, "final_output": answer1})
    print(f"[Memory] {save1['execution_trace'][-1].split(': ', 1)[1]}")
    print("\n[Long-term memory now contains]")
    for record in engine.list_memories("demo_user"):
        print(f"  - ({record.memory_type.value}) {textwrap.shorten(record.text, 110)}")

    # ------------------------------------------------------------------ session 2
    section("SESSION 2 - a new session, different wording")
    q2 = "What did we conclude about GPU makers' data centre growth?"
    print(f"[User] {q2}")

    print("\n---------------- WITHOUT MEMORY ----------------")
    print("Supervisor memory block: (empty)")
    print("Preferences            : {}")
    print("\nResponse:\n" + render_answer({}))

    s2 = state(q2, "session_002")
    recall2 = memory_recall_node(s2)
    print("\n---------------- WITH MEMORY -------------------")
    print("Supervisor memory block:\n" + textwrap.indent(build_supervisor_memory_block({**s2, **recall2}), "  "))
    print(f"\nPreferences            : {recall2['user_preferences']}")
    print("\nResponse:\n" + render_answer(recall2["user_preferences"]))

    # ------------------------------------------------------------------ isolation & forget
    section("PRIVACY CHECKS")
    other = memory_recall_node({**s2, "user_id": "another_user"})
    print(f"[Isolation] another_user sees: {other['recalled_context']!r}")
    removed = engine.forget_user("demo_user")
    print(f"[Forget me] removed {removed} records for 'demo_user'; remaining = {len(engine.list_memories('demo_user'))}")

    section("Developer 3 Module Summary")
    print("1. Short-term session buffer       : ACTIVE")
    print("2. Long-term ChromaDB memory        : ACTIVE (per-user partitioned)")
    print("3. all-MiniLM-L6-v2 embeddings      : ACTIVE (ONNX, no HuggingFace / PyTorch)")
    print("4. Automatic memory extraction      : ACTIVE (rules; LLM when OPENAI_API_KEY is set)")
    print("5. Cosine-similarity RAG retrieval  : ACTIVE (threshold + recency + MMR + token budget)")
    print("6. Dedup / PII masking / cap / TTL  : ACTIVE")
    line()
    set_default_engine(None)


if __name__ == "__main__":
    main()
