"""
Retrieval quality evaluation for the Memory Engine (Developer 3 - Rajarshi).

20 labelled queries, each with exactly one memory that should come back.
None of the queries reuse the memory's wording, so this measures semantic
recall with the real all-MiniLM-L6-v2 model, not keyword matching.

Metrics:
  Hit@k = share of queries whose expected memory is in the top k
  MRR   = mean of 1 / rank of the expected memory (0 if not returned)

Run:  .venv\\Scripts\\python eval\\retrieval_eval.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_engine import MemoryConfig, MemoryEngine, MemoryType  # noqa: E402
from memory_engine.extractor import RuleBasedExtractor  # noqa: E402

USER = "eval_user"

MEMORIES: dict[str, tuple[str, MemoryType]] = {
    "employer": ("User works at Acme Corp as a data analyst", MemoryType.FACT),
    "city": ("User is based in Pune, India", MemoryType.FACT),
    "sector": ("User mainly tracks the semiconductor sector", MemoryType.FACT),
    "portfolio": ("User holds NVIDIA and AMD shares in a long-term portfolio", MemoryType.FACT),
    "risk": ("User has a low risk appetite and avoids speculative stocks", MemoryType.FACT),
    "language": ("User codes mostly in Python with pandas", MemoryType.FACT),
    "exam": ("User is preparing for the CFA level 2 exam", MemoryType.FACT),
    "team": ("User manages a team of four junior analysts", MemoryType.FACT),
    "nvda_growth": ("Past question: NVIDIA data center growth | Conclusion: NVIDIA data center revenue rose 36% from Q1 to Q3", MemoryType.CONCLUSION),
    "amd_growth": ("Past question: AMD growth | Conclusion: AMD data center sales expanded 52% from a smaller base", MemoryType.CONCLUSION),
    "tsmc": ("Past question: supply chain risk | Conclusion: TSMC CoWoS packaging capacity is the main bottleneck for AI chips", MemoryType.CONCLUSION),
    "rates": ("Past question: interest rates | Conclusion: the Fed held rates steady and signalled two cuts next year", MemoryType.CONCLUSION),
    "ev": ("Past question: electric vehicles | Conclusion: Tesla margins fell as EV price competition in China intensified", MemoryType.CONCLUSION),
    "oil": ("Past question: crude oil | Conclusion: Brent crude stayed near 80 dollars due to OPEC production cuts", MemoryType.CONCLUSION),
    "inflation": ("Past question: India inflation | Conclusion: India CPI inflation eased to 4.8 percent, inside the RBI target band", MemoryType.CONCLUSION),
    "cloud": ("Past question: cloud market | Conclusion: Microsoft Azure grew faster than AWS last quarter", MemoryType.CONCLUSION),
}

QUERIES: list[tuple[str, str]] = [
    ("Which company does the user work for?", "employer"),
    ("Where does the user live?", "city"),
    ("What industry does the user follow?", "sector"),
    ("Which stocks does the user own?", "portfolio"),
    ("How cautious is the user with investments?", "risk"),
    ("What programming language does the user use?", "language"),
    ("Is the user studying for any certification?", "exam"),
    ("Does the user lead people?", "team"),
    ("How fast did NVIDIA's data centre business grow?", "nvda_growth"),
    ("What did we find about AMD's server revenue?", "amd_growth"),
    ("What limits AI chip production?", "tsmc"),
    ("What is the central bank doing with interest rates?", "rates"),
    ("What happened to Tesla's profitability?", "ev"),
    ("Where are oil prices?", "oil"),
    ("How is inflation in India?", "inflation"),
    ("Who is winning in cloud computing?", "cloud"),
    ("Remind me of the user's employer", "employer"),
    ("Chip packaging constraints", "tsmc"),
    ("Petroleum price outlook", "oil"),
    ("Semiconductor shares the user has invested in", "portfolio"),
]


def build_engine() -> MemoryEngine:
    config = MemoryConfig(persist_dir=None, collection_name=f"eval_{uuid.uuid4().hex[:8]}", min_similarity=0.0)
    engine = MemoryEngine(config=config, extractor=RuleBasedExtractor())
    for text, memory_type in MEMORIES.values():
        engine.remember(USER, text, memory_type)
    return engine


def run_eval(engine: MemoryEngine, k: int = 5) -> dict:
    text_to_key = {text: key for key, (text, _) in MEMORIES.items()}
    hits1 = hits3 = 0
    reciprocal_ranks = []
    rows = []
    for query, expected in QUERIES:
        results = engine.retriever.retrieve(USER, query, top_k=k)
        keys = [text_to_key.get(r.record.text) for r in results]
        rank = keys.index(expected) + 1 if expected in keys else 0
        hits1 += rank == 1
        hits3 += 1 <= rank <= 3
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        top_similarity = results[0].similarity if results else 0.0
        rows.append((query, expected, rank, keys[0] if keys else None, top_similarity))
    n = len(QUERIES)
    return {
        "queries": n,
        "hit@1": hits1 / n,
        "hit@3": hits3 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "rows": rows,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    metrics = run_eval(build_engine())
    print(f"{'Query':<55} {'Expected':<12} {'Rank':>4}  {'Top-1':<12} {'Sim':>5}")
    print("-" * 95)
    for query, expected, rank, top, sim in metrics["rows"]:
        print(f"{query:<55} {expected:<12} {rank or '-':>4}  {str(top):<12} {sim:>5.2f}")
    print("-" * 95)
    print(f"Hit@1 = {metrics['hit@1']:.2f}   Hit@3 = {metrics['hit@3']:.2f}   MRR = {metrics['mrr']:.3f}   (n={metrics['queries']})")


if __name__ == "__main__":
    main()
