from src.worker_agents.research_agent import ResearchAgent
from src.worker_agents.schemas import ResearchRequest


agent = ResearchAgent()

request = ResearchRequest(
    query="LangGraph",
    max_results=5,
)

result = agent.run(request)

print("Success:", result.success)
print("Sources:", len(result.sources))
print("Error:", result.error)

for source in result.sources:
    print()
    print("Title:", source.title)
    print("URL:", source.url)