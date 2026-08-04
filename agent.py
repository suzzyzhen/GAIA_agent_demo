"""
agent.py
Builds the LangGraph agent: wires an LLM (Google Gemini / Groq / HuggingFace)
together with the tools defined in tools.py.
"""

from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from tools import tools
import re


load_dotenv()
# ======================================================================================
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
HUGGINGFACE_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

SYSTEM_PROMPT_PATH = "system_prompt.txt"

# Read once at import time instead of on every assistant invocation.
with open(SYSTEM_PROMPT_PATH, "r") as f:
    _SYSTEM_PROMPT = f.read()


def _build_llm(provider: str):
    if provider == "google":
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    elif provider == "groq":
        return ChatGroq(model=GROQ_MODEL, temperature=0)
    elif provider == "huggingface":
        return ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                model=HUGGINGFACE_MODEL,
                temperature=0,
            )
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")


def build_graph(provider: str = "google"):
    """Build the agent graph."""
    llm = _build_llm(provider)
    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(content=_SYSTEM_PROMPT)

    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)

    # Nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile()

    return react_graph

def extract_final_answer(text: str) -> str:
    """Pull the text following 'FINAL ANSWER:' out of the agent's raw response,
    falling back to the full response if the marker isn't present."""
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def run_agent(graph, question: str, recursion_limit: int = 50) -> str:
    """Invoke the graph on a single question and return the cleaned final answer."""
    from langchain_core.messages import HumanMessage

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": recursion_limit},
    )
    raw_answer = result["messages"][-1].content
    return extract_final_answer(raw_answer)