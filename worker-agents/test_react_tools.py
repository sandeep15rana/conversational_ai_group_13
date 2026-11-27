from src.worker_agents.react_agent import get_worker_tools


tools = get_worker_tools()

print("Number of tools:", len(tools))

for tool in tools:
    print("Tool:", tool.name)
    print("Description:", tool.description)
    print()
