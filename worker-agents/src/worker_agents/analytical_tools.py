from typing import Any

import pandas as pd

from .exceptions import AnalyticalAgentError
from .schemas import AnalysisRequest, AnalysisResponse


def _records(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict(
            orient="records"
        )

    if isinstance(value, pd.Series):
        return value.where(pd.notna(value), None).to_dict()

    return value


class PandasAnalysisTool:
    def execute(
        self,
        request: AnalysisRequest,
    ) -> AnalysisResponse:
        try:
            frame = pd.DataFrame(request.data)

            if frame.empty:
                raise AnalyticalAgentError(
                    "The analytical dataset is empty."
                )

            operation = request.operation
            parameters = request.parameters

            if operation == "summary":
                result = {
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "missing_values": (
                        frame.isna().sum().to_dict()
                    ),
                    "data_types": {
                        column: str(dtype)
                        for column, dtype in frame.dtypes.items()
                    },
                }

            elif operation == "describe":
                result = _records(
                    frame.describe(include="all").transpose()
                )

            elif operation == "groupby":
                group_column = parameters["group_column"]
                value_column = parameters["value_column"]
                aggregation = parameters.get(
                    "aggregation",
                    "sum",
                )

                if aggregation not in {
                    "sum",
                    "mean",
                    "min",
                    "max",
                    "count",
                }:
                    raise AnalyticalAgentError(
                        f"Unsupported aggregation: {aggregation}"
                    )

                grouped = (
                    frame.groupby(group_column)[value_column]
                    .agg(aggregation)
                    .reset_index()
                )
                result = _records(grouped)

            elif operation == "filter":
                column = parameters["column"]
                operator = parameters["operator"]
                value = parameters["value"]

                operators = {
                    "eq": frame[column] == value,
                    "ne": frame[column] != value,
                    "gt": frame[column] > value,
                    "ge": frame[column] >= value,
                    "lt": frame[column] < value,
                    "le": frame[column] <= value,
                }

                if operator not in operators:
                    raise AnalyticalAgentError(
                        f"Unsupported filter operator: {operator}"
                    )

                result = _records(frame[operators[operator]])

            elif operation == "correlation":
                numeric_frame = frame.select_dtypes(
                    include="number"
                )
                result = _records(numeric_frame.corr())

            else:
                raise AnalyticalAgentError(
                    f"Unsupported operation: {operation}"
                )

            return AnalysisResponse(
                operation=operation,
                result=result,
                row_count=len(frame),
                columns=list(frame.columns),
            )

        except KeyError as exc:
            raise AnalyticalAgentError(
                f"Required parameter or column is missing: {exc}"
            ) from exc
        except AnalyticalAgentError:
            raise
        except Exception as exc:
            raise AnalyticalAgentError(
                f"Data analysis failed: {exc}"
            ) from exc