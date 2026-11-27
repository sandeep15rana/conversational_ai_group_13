from src.worker_agents.analytical_agent import AnalyticalAgent
from src.worker_agents.research_agent import ResearchAgent
from src.worker_agents.schemas import (
    AnalysisRequest,
    ResearchRequest,
    SynthesisRequest,
)
from src.worker_agents.synthesis_agent import SynthesisAgent


def main() -> None:
    research_agent = ResearchAgent()
    analytical_agent = AnalyticalAgent()
    synthesis_agent = SynthesisAgent()

    research = research_agent.run(
        ResearchRequest(
            query="Current applications of multi-agent AI systems",
            max_results=3,
        )
    )

    sales_data = [
        {"region": "North", "sales": 120},
        {"region": "South", "sales": 180},
        {"region": "North", "sales": 130},
        {"region": "South", "sales": 220},
    ]

    analysis = analytical_agent.run(
        AnalysisRequest(
            operation="groupby",
            data=sales_data,
            parameters={
                "group_column": "region",
                "value_column": "sales",
                "aggregation": "sum",
            },
        )
    )

    final_response = synthesis_agent.run(
        SynthesisRequest(
            original_query=(
                "Explain multi-agent AI and summarize the "
                "regional sales data."
            ),
            research_findings=research,
            analytical_findings=analysis,
        )
    )

    print(final_response.markdown)


if __name__ == "__main__":
    main()