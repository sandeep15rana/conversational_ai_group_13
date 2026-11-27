# Memory Engine (Developer 3 – Rajarshi)

Dual-layer memory for the Group 13 multi-agent system.
- **Short-term:** a per-session conversation buffer.
- **Long-term:** ChromaDB + `all-MiniLM-L6-v2` embeddings, partitioned per user.

👉 **Integrating it?** Read [INTEGRATION.md](INTEGRATION.md).

## Files

| File | What it does |
|---|---|
| `src/memory_engine/config.py` | All settings, validated, overridable via `MEMORY_*` env vars |
| `src/memory_engine/exceptions.py` | One error family (`MemoryEngineError`) |
| `src/memory_engine/schemas.py` | Pydantic models: memory record, candidate, recall result |
| `src/memory_engine/privacy.py` | PII masking, prompt-injection filter, pseudonymous user ids |
| `src/memory_engine/embeddings.py` | all-MiniLM-L6-v2 (ONNX via ChromaDB, no HuggingFace/PyTorch), cosine formula |
| `src/memory_engine/chunking.py` | Sentence-aware chunks ≤ 200 tokens with 1-sentence overlap |
| `src/memory_engine/short_term.py` | Session buffer: last N turns, TTL, LRU, scratchpad, thread-safe |
| `src/memory_engine/long_term.py` | ChromaDB store: per-user isolation, dedupe, conflict resolution, cap, forget-me |
| `src/memory_engine/extractor.py` | Picks preferences, facts and conclusions (rules; LLM if a key is set) |
| `src/memory_engine/retriever.py` | RAG: profile + cosine search → threshold → recency → MMR → token budget |
| `src/memory_engine/engine.py` | `MemoryEngine`: wires everything together |
| `src/memory_engine/nodes.py` | The two LangGraph nodes, scratchpad helpers, ablation baseline, helpers for other teams |
| `tests/` | 229 tests: unit, validation, real-model, review regressions, integration with Sandeep's graph |
| `eval/retrieval_eval.py` | Retrieval quality on 20 labelled queries (Hit@k, MRR) |
| `demo_memory.py` | "With memory vs without memory" demo |

## Quick start

From inside `memory-engine/`:

```powershell
py -3.12 -m venv .venv
```
```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```
```powershell
.venv\Scripts\python -m pip install -e .
```
```powershell
.venv\Scripts\python demo_memory.py
```

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```
Fast tests only (no model download):
```powershell
.venv\Scripts\python -m pytest -q -m "not integration"
```

## Results

| Metric | Value |
|---|---|
| Tests | 229 passed |
| Line coverage | 95% |
| Retrieval Hit@1 / Hit@3 / MRR | 0.80 / 1.00 / 0.90 |
