from .analytical_tools import PandasAnalysisTool
from .exceptions import AnalyticalAgentError
from .schemas import AnalysisRequest, AnalysisResponse


class AnalyticalAgent:
    name = "analytical"

    def __init__(
        self,
        tool: PandasAnalysisTool | None = None,
    ) -> None:
        self.tool = tool or PandasAnalysisTool()

    def run(self, request: AnalysisRequest) -> AnalysisResponse:
        try:
            return self.tool.execute(request)
        except AnalyticalAgentError as exc:
            return AnalysisResponse(
                operation=request.operation,
                result=None,
                row_count=len(request.data),
                columns=(
                    list(request.data[0].keys())
                    if request.data
                    else []
                ),
                success=False,
                error=str(exc),
            )