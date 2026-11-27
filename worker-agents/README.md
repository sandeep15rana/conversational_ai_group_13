Roles & Responsibilities Segregation

💻 Developer 1: Orchestration Engine, Planning & State Graph Lead
	Owner: Sandeep Rana
	Syllabus Link: Sessions L1 & L9 | Labs 9 & 10 (Agent Architectures, Planning, LLM Orchestration).
	Core Responsibilities:
	Build the master multi-agent workflow using LangGraph (StateGraph).
	Implement the Supervisor / Planner Agent that analyzes user requests and creates an execution DAG (Directed Acyclic Graph).
	Implement conditional edge routing (e.g., should we search the web, calculate with Python, or synthesize the final answer?).
	Implement state persistence, recursion limiters (preventing runaway loops), and error recovery fallbacks.
	Deliverable to Doc Lead: State-machine control flow logic, supervisor prompt templates, and sample routing decisions.
________________________________________
💻 Developer 2: Specialized Worker Agents & ReAct Tool Execution Lead
	Owner: Pooja Gholap
	Syllabus Link: Session L4 | Labs 1 & 4 (Structured Outputs, Function Calling, ReAct Framework).
	Core Responsibilities:
	Implement the ReAct (Thought → Action → Observation) reasoning loop for sub-agents.
	Research Agent: Integrate native function calling with DuckDuckGo Search API or Tavily to fetch real-time external data.
	Analytical Agent: Sandboxed Python REPL tool using pandas to execute calculations, percentage growth, and financial formulas.
	Synthesis Agent: Formats and merges findings from other agents into a clean markdown briefing.
	Define strict Pydantic schemas for tool inputs/outputs and handle execution exceptions/retries.
	Deliverable to Doc Lead: Tool schemas, raw execution logs showing the step-by-step Thought → Action → Observation trace.
________________________________________
💻 Developer 3: Dual-Layer Memory & Context Engine Lead
	Owner: Rajarshi
	Syllabus Link: Session L6 | MemGPT Concepts (Short-Term vs. Long-Term Memory, Hybrid SQL + Vector).
	Core Responsibilities:
	Short-Term Memory: Manage session thread buffers to maintain multi-turn dialogue context and intermediate agent scratchpads.
	Long-Term Episodic Memory: Integrate ChromaDB or FAISS with sentence-transformers embeddings to persist user profiles, past conversation conclusions, and preferred report formats across sessions.
	Memory Extraction Pipeline: Build an automatic post-session consolidation agent that identifies salient facts/user preferences and writes them into the vector store.
	Semantic Memory Retrieval: Implement cosine similarity search to retrieve relevant historical memories and inject them into the Supervisor's prompt.
	Deliverable to Doc Lead: Memory architecture explanation, cosine similarity formulas, and a concrete comparison showing "Response with Memory vs. Response without Memory".
________________________________________
💻 Developer 4: Security Guardrails & Quantitative Evaluation Lead
	Owner: Harshavardhan M
	Syllabus Link: Sessions L10 & L12 | Labs 7 & 8 (Security, Adversarial Robustness, RAG/Agent Evaluation, LLM-as-a-Judge).
	Core Responsibilities:
	Security Guardrails:
	Input Guard: Detect direct and indirect prompt injection attempts.
	Privacy Guard: Implement PII scrubbing (anonymizing emails, phone numbers, and API keys).
	Output Guard: Check that the final answer is grounded in agent findings (hallucination defense).
	Quantitative Benchmarking Suite:
	Curate a test dataset of 20 challenging multi-step queries.
	Implement an LLM-as-a-Judge evaluator to score Task Completion Rate, Tool Selection Accuracy, and Groundedness.
	Measure and profile token economics (cost per task) and end-to-end execution latency.
	Ablation Study: Run baseline comparison: Single ReAct Agent vs. Proposed Multi-Agent System.
	Deliverable to Doc Lead: Attack mitigation rates (before vs. after guardrails), quantitative evaluation tables, and radar/bar chart data.
________________________________________
2. The Documentation & Synthesis Lead (Member 5)
	Owner: Hemlata Mingwal
📝 Member 5: Chief Technical Documentarian & System Architect
	Role Value: Ensures the project receives top marks by transforming raw code and logs into a publication-ready, deeply technical, 8–10 page report.
	Core Responsibilities:
	System Architecture & Visuals:
	Create professional system diagrams (Overall Architecture, LangGraph State Machine, ReAct Sequence Diagram, Dual-Memory Flowchart).
	Theoretical Literature & Grounding:
	Write the technical background comparing monolithic LLMs vs. Multi-Agent systems, citing relevant papers from the handout (Yao et al., 2023 for ReAct; Packer et al., 2023 for MemGPT).
	Experimental Results & Tabular Synthesis:
	Format Developer 4’s benchmark data into clean comparative tables and graphs.
	Document concrete failure modes (e.g., when an agent timed out, selected the wrong tool, or was blocked by a guardrail).
	Report Authoring & Final Quality Control:
	Compile the complete report in Word/LaTeX and export to PDF.
	Ensure academic tone, proper section numbering, references, code snippet formatting, and zero plagiarism.
 

