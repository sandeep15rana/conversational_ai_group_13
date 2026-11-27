from src.worker_agents.analytical_agent import AnalyticalAgent
from src.worker_agents.schemas import AnalysisRequest


data = [
    {"region": "North", "sales": 120},
    {"region": "South", "sales": 180},
    {"region": "North", "sales": 130},
    {"region": "South", "sales": 220},
]

request = AnalysisRequest(
    operation="groupby",
    data=data,
    parameters={
        "group_column": "region",
        "value_column": "sales",
        "aggregation": "sum",
    },
)

agent = AnalyticalAgent()
result = agent.run(request)

print("Success:", result.success)
print("Operation:", result.operation)
print("Rows:", result.row_count)
print("Columns:", result.columns)
print("Result:", result.result)
print("Error:", result.error)