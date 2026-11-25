# 🚀 Group 13 – Project Overview & Work Distribution

Hi Everyone,

As discussed, we have divided the project into five major responsibilities to ensure parallel development and smooth integration.

---

## 1. Project Overview

We are building a **Multi-Agent AI System using LangGraph**, where multiple specialized agents collaborate to understand user requests, execute tasks, retrieve information, perform analysis, and generate a final response.

The overall workflow consists of:

- **Input Guardrail:** Checks malicious prompts, prompt injection attempts, and sensitive information such as API keys, emails, and phone numbers.
- **Memory Engine:** Retrieves relevant long-term user context and preferences.
- **Supervisor / Planner Agent:** Understands the request, breaks it into subtasks, and decides which agents/tools should execute them.
- **Research Agent:** Fetches real-time information using DuckDuckGo or Tavily.
- **Analytical Agent:** Executes Python-based calculations using `pandas`.
- **Synthesis Agent:** Combines the findings into a structured response.
- **Output Guardrail:** Checks the final response for hallucinations and verifies that it is grounded in the findings.
- **Observability & Evaluation:** Tracks token usage, execution latency, loop count, task completion, and evaluation scores.

---

## 2. Individual Responsibilities

### 💻 Developer 1 – Sandeep Rana

**Orchestration Engine, Planning & State Graph**

- Build the master multi-agent workflow using LangGraph `StateGraph`.
- Implement Supervisor / Planner Agent.
- Decompose user requests into an execution DAG.
- Implement conditional routing between agents.
- Handle state persistence, recursion limits, and error recovery.

### 💻 Developer 2 – Pooja Gholap

**Specialized Worker Agents & ReAct Tool Execution**

- Implement the ReAct reasoning loop (`Thought → Action → Observation`).
- Develop Research Agent using DuckDuckGo / Tavily.
- Develop Analytical Agent using Python REPL and `pandas`.
- Develop Synthesis Agent for structured Markdown responses.
- Define Pydantic schemas and implement exception handling/retries.

### 💻 Developer 3 – Rajarshi

**Dual-Layer Memory & Context Engine**

- Implement short-term conversational memory.
- Implement long-term memory using ChromaDB / FAISS.
- Integrate `sentence-transformers` embeddings.
- Develop automatic memory extraction and persistence.
- Implement semantic memory retrieval using cosine similarity.

### 💻 Developer 4 – Harshavardhan M

**Security Guardrails & Quantitative Evaluation**

- Implement input guardrails for prompt injection.
- Implement PII detection and sanitization.
- Implement output validation and hallucination checks.
- Prepare a dataset of 20 challenging multi-step queries.
- Implement LLM-as-a-Judge evaluation.
- Measure task completion, tool selection accuracy, groundedness, token consumption, and latency.
- Conduct an ablation study comparing Single ReAct Agent vs Multi-Agent System.

### 📝 Developer 5 – Hemlata Mingwal

**Documentation & Technical Report**

- Prepare overall system architecture diagrams.
- Create LangGraph state machine, ReAct sequence, and memory flow diagrams.
- Document theoretical concepts and relevant research papers.
- Consolidate implementation details and experimental results.
- Prepare comparison tables, graphs, and failure-mode analysis.
- Compile the final 8–10 page technical report and export it to PDF.

---

## 3. GitHub Collaboration & Development Guidelines

To maintain clean code management and avoid conflicts, everyone needs to follow these steps:

### Step 1: Clone the repository

GitHub Repository: [conversational_ai_group_13](https://github.com/sandeep15rana/conversational_ai_group_13)

```bash
git clone https://github.com/sandeep15rana/conversational_ai_group_13.git
cd conversational_ai_group_13
```

### Step 2: Create your own branch

Everyone must create a separate branch for their assigned responsibility.

Example:

```bash
git checkout -b feature/orchestration
```

**Suggested branch names:**

- `feature/orchestration`
- `feature/worker-agents`
- `feature/memory-engine`
- `feature/guardrails-evaluation`
- `feature/documentation`

### Step 3: Develop and commit your changes

Work independently within your assigned module.

```bash
git add .
git commit -m "Implement assigned module"
git push -u origin feature/your-branch-name
```

Once the work is ready, we can review and merge individual branches into the `main` branch.

---

## 4. Parallel Development & Integration Strategy

Since some components depend on other members' implementations, we should not wait for all modules to be completed before starting development.

- Each member should initially develop their assigned module independently.
- Use mock data, dummy responses, placeholder functions, and mock interfaces wherever dependent components are not yet available.
- Perform standalone testing of individual modules before integration.
- Define expected input/output formats and interfaces between dependent modules.
- Once individual modules are ready, we will integrate them into the master workflow.
- Resolve integration issues together and perform end-to-end testing.

> **Example:** The Orchestration Engine can initially use mock Research, Analytical, and Memory agents. Similarly, the Research and Analytical agents can be tested independently without waiting for the Supervisor implementation.

---

## 5. Final Objective

Our goal is to develop a working, integrated Multi-Agent AI System with proper security, memory, tool execution, evaluation, and observability, supported by technical documentation and experimental results.

Let's focus on completing our individual modules independently first, maintain regular Git commits, and then integrate everything systematically.

Please let me know if anyone has questions or suggestions regarding the task allocation or development approach. 🚀
