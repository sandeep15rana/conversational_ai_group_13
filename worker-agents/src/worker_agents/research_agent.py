from .exceptions import ResearchAgentError
from .research_tools import DuckDuckGoResearchTool
from .schemas import ResearchRequest, ResearchResponse


class ResearchAgent:
    name = "research"

    def __init__(
        self,
        tool: DuckDuckGoResearchTool | None = None,
    ) -> None:
        self.tool = tool or DuckDuckGoResearchTool()

    def run(
        self,
        request: ResearchRequest,
    ) -> ResearchResponse:

        try:
            return self.tool.search(request)

        except ResearchAgentError as exc:
            return ResearchResponse(
                query=request.query,
                provider="duckduckgo",
                success=False,
                error=str(exc),
            )