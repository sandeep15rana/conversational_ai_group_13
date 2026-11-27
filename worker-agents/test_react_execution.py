from src.worker_agents.react_agent import (
    analyze_records,
    web_research,
)


print("========== RESEARCH TOOL ==========")

research_result = web_research.invoke(
    {
        "query": "LangGraph",
        "max_results": 3,
    }
)

print("Success:", research_result["success"])
print("Sources:", len(research_result["sources"]))

for source in research_result["sources"]:
    print("-", source["title"])


print()
print("========== ANALYTICAL TOOL ==========")

sales_data = [
    {"region": "North", "sales": 120},
    {"region": "South", "sales": 180},
    {"region": "North", "sales": 130},
    {"region": "South", "sales": 220},
]

analysis_result = analyze_records.invoke(
    {
        "operation": "groupby",
        "data": sales_data,
        "parameters": {
            "group_column": "region",
            "value_column": "sales",
            "aggregation": "sum",
        },
    }
)

print("Success:", analysis_result["success"])
print("Result:", analysis_result["result"])