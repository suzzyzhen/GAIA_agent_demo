from agent import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph(provider="huggingface") 
result = graph.invoke(
    {"messages": [HumanMessage(content="What is the capital of France?")]},
    config={"recursion_limit": 50},
)
print(result["messages"][-1].content)