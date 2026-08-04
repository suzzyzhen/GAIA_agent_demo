from agent import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph(provider="huggingface") 
result = graph.invoke(
    {"messages": [HumanMessage(content="What is the answer of 5 + 3?")]},
    config={"recursion_limit": 50},
)
print(result["messages"][-1].content)