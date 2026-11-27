from types import SimpleNamespace

from src.worker_agents.schemas import (
    AnalysisResponse,
    ResearchResponse,
    ResearchSource,
    SynthesisRequest,
)
from src.worker_agents.synthesis_agent import SynthesisAgent


class MockLLM:
    def invoke(self, prompt):
        return SimpleNamespace(
            content="""
# Analysis Summary

## Research Findings

LangGraph is a framework used for building
stateful agent workflows.

## Data Analysis

Regional sales totals:

- North: 250
- South: 400

South has higher total sales than North.

## Conclusion

The research and analytical agents successfully
provided information for synthesis.
"""
        )


research = ResearchResponse(
    query="LangGraph",
    answer=None,
    provider="duckduckgo",
    sources=[
        ResearchSource(
            title="LangGraph Documentation",
            url="https://docs.langchain.com/",
            content="LangGraph documentation.",
            score=None,
        )
    ],
)


analysis = AnalysisResponse(
    operation="groupby",
    result=[
        {"region": "North", "sales": 250},
        {"region": "South", "sales": 400},
    ],
    row_count=4,
    columns=["region", "sales"],
)


request = SynthesisRequest(
    original_query=(
        "Explain LangGraph and summarize regional sales."
    ),
    research_findings=research,
    analytical_findings=analysis,
)


agent = SynthesisAgent(model=MockLLM())
result = agent.run(request)


print("Success:", result.success)
print()
print(result.markdown)
print()
print("Citations:", result.citations)