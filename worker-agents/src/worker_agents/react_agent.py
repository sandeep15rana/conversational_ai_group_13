from typing import Any

from langchain_core.tools import tool

from .analytical_agent import AnalyticalAgent
from .research_agent import ResearchAgent
from .schemas import AnalysisRequest, ResearchRequest


research_agent = ResearchAgent()
analytical_agent = AnalyticalAgent()


@tool
def web_research(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Search the web for information.

    Use this tool when the task requires external
    or up-to-date information.
    """

    request = ResearchRequest(
        query=query,
        max_results=max_results,
    )

    result = research_agent.run(request)

    return result.model_dump(mode="json")


@tool
def analyze_records(
    operation: str,
    data: list[dict],
    parameters: dict | None = None,
) -> dict[str, Any]:
    """
    Analyze structured data using pandas.

    Supported operations include summary, describe,
    groupby, filter, and correlation.
    """

    request = AnalysisRequest(
        operation=operation,
        data=data,
        parameters=parameters or {},
    )

    result = analytical_agent.run(request)

    return result.model_dump(mode="json")


TOOLS = [
    web_research,
    analyze_records,
]


def get_worker_tools():
    """Return the tools available to the ReAct agent."""

    return TOOLS