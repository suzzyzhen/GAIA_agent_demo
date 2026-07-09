from tools import (
    add, subtract, multiply, divide, power, modulus, square_root,
    web_search,
    wikipedia_search,
    arxiv_search,
    pdf_reader,
    spreadsheet_reader,
    image_ocr,
    code_file_interpreter, 
    analyze_image
)
import wikipediaapi
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langgraph.graph.message import add_messages


# GROQ_API_KEY = os.environ["GROQ_API_KEY"] 

# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# HF_TOKEN = os.environ["HF_TOKEN"]


tools = [
    web_search,
    wikipedia_search,
    arxiv_search,
    add, subtract, multiply, divide, power, modulus, square_root,
    pdf_reader,
    spreadsheet_reader,
    image_ocr,
    code_file_interpreter, 
    analyze_image,
]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph(provider: str = "google"):
    """Build the graph"""
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            # huggingfacehub_api_token=os.environ["HF_TOKEN"],
            temperature=0,
          ),
            verbose=False,
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: AgentState):
        """Assistant node"""
        with open('system_prompt.txt', 'r') as f:
            system_prompt = f.read()
        sys_msg = SystemMessage(content=system_prompt)

        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))


    # Edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile()

    return react_graph