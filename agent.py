# from tools import (
#     add, subtract, multiply, divide, power, modulus, square_root,
#     web_search,
#     wikipedia_search,
#     arxiv_search,
#     pdf_reader,
#     spreadsheet_reader,
#     image_ocr,
#     code_file_interpreter, 
#     analyze_image
# )
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
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from ddgs import DDGS 
import re


# GROQ_API_KEY = os.environ["GROQ_API_KEY"] 

# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# HF_TOKEN = os.environ["HF_TOKEN"]

# ======================================================================================
gemini_model = "gemini-2.5-flash"
groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
huggingFace_model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


from langchain_core.tools import tool
import os
import arxiv
import wikipediaapi
import pdfplumber
from pdf2image import convert_from_path
import pandas as pd
import pytesseract
# from PIL import Image
import PIL.Image
import subprocess
from langchain_tavily import TavilySearch
from typing import Optional

# ========================Calculator Tools========================
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b and return the result. Raises an error if b is 0."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def power(a: float, b: float) -> float:
    """Raise a to the power of b and return the result."""
    return a ** b

@tool
def modulus(a: float, b: float) -> float:
    """Return the remainder of a divided by b."""
    return a % b

@tool
def square_root(a: float) -> float:
    """Return the square root of a. Raises an error if a is negative."""
    if a < 0:
        raise ValueError("Cannot take square root of a negative number.")
    return a ** 0.5

#  ========================Search Tools========================

@tool
def web_search(query: str) -> str:
    """Search the web for a query and return the top results including
    title, URL, and content snippet for each.
    Args:
        query: The search query."""

    docs = []

    # Try DuckDuckGo first
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        docs = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
            for r in results
        ]
    except Exception:
        docs = []

    # Fall back to Tavily if DDG failed or returned nothing
    if not docs:
        search = TavilySearch(max_results=3, api_key=os.environ.get("TAVILY_API_KEY"))
        responses = search.invoke(query)

        if isinstance(responses, dict):
            raw_docs = responses.get("results", [])
        elif isinstance(responses, list):
            raw_docs = responses
        else:
            raw_docs = []

        docs = [
            {"title": d.get("title", ""), "url": d.get("url", ""), "content": d.get("content", "")}
            for d in raw_docs
        ]

    if not docs:
        return "No results found."

    return "\n\n".join(
        f"[{i}]\n"
        f"  Title: {doc['title']}\n"
        f"  URL: {doc['url']}\n"
        f"  Content: {doc['content']}"
        for i, doc in enumerate(docs, start=1)
    )

@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers matching the query and return
    titles, authors, and abstracts of the top matches."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=2)
    results = client.results(search)

    formatted = []
    for result in results:
        formatted.append(
            f"Title: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Published: {result.published.date()}\n"
            f"Summary: {result.summary[:1000]}\n"
            f"URL: {result.entry_id}"
        )
    return "\n\n---\n\n".join(formatted) if formatted else "No results found."

wiki_client = wikipediaapi.Wikipedia(
    user_agent="MyGAIAAgent/1.0 (myemail@example.com)",
    language="en"
)

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia. REQUIRED: you must provide a non-empty 'query' string
    parameter containing the search term, e.g. query='Alan Turing'."""
    
    page = wiki_client.page(query)
    if not page.exists():
        return f"No Wikipedia page found for '{query}'."
    return page.summary[:2000]



#  ========================Files Tools========================
@tool
def pdf_reader(file_path: str) -> str:
    """Extract text from a PDF file at the given local file path.
    Falls back to OCR automatically if the PDF is scanned/image-based."""
    text_parts = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    extracted_text = "\n".join(text_parts).strip()

    if len(extracted_text) < 20:
        images = convert_from_path(file_path)
        ocr_parts = [pytesseract.image_to_string(img) for img in images]
        extracted_text = "\n".join(ocr_parts).strip()

    return extracted_text if extracted_text else "No text could be extracted from this PDF."

@tool
def spreadsheet_reader(
    file_path: str,
    sheet_name: Optional[str] = None,
) -> str:
    """Read a CSV or Excel file.

    Args:
        file_path: Path to a CSV or Excel file.
        sheet_name: Name of the Excel sheet. If omitted, all sheets are read.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_markdown(index=False)

    if sheet_name is not None:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df.to_markdown(index=False)

    sheets = pd.read_excel(file_path, sheet_name=None)
    return "\n\n---\n\n".join(
        f"## Sheet: {name}\n\n{df.to_markdown(index=False)}"
        for name, df in sheets.items()
    )
    
@tool
def image_ocr(file_path: str) -> str:
    """Extract any visible text from an image file using OCR.
    Best for screenshots, scanned documents, charts with labels, or text-heavy images."""
    img = PIL.Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text.strip() if text.strip() else "No text found in image."


@tool
def code_file_interpreter(file_path: str, mode: str = "execute") -> str:
    """Read or execute a code file at the given file path.
    mode='execute': runs the file (Python only) and returns stdout/stderr.
    mode='read': returns the raw source code as text, for inspection/reasoning
    without running it."""
    
    if mode == "read":
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    elif mode == "execute":
        if not file_path.endswith(".py"):
            return "Error: execution is only supported for .py files. Use mode='read' for other file types."
        try:
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            error = result.stderr.strip()
            if error:
                return f"STDOUT:\n{output}\n\nSTDERR:\n{error}"
            return output if output else "Code executed successfully with no output."
        except subprocess.TimeoutExpired:
            return "Error: code execution timed out after 30 seconds."
        except Exception as e:
            return f"Error executing file: {e}"

    else:
        return f"Unknown mode: {mode}. Use 'execute' or 'read'."
    

@tool
def audio_transcriber(file_path: str) -> str:
    """Transcribe an audio file (mp3, wav, m4a, etc.) to text.
    Use this for any question that references an audio recording,
    lecture, voicemail, or similar attached audio file."""
    _whisper_model = whisper.load_model("base")
    result = _whisper_model.transcribe(file_path)
    return result["text"].strip()


@tool
def youtube_transcript(url: str, chars: int = 10_000) -> str:
    """Fetch full YouTube transcript (first *chars* characters)."""
    video_id_match = re.search(r"[?&]v=([A-Za-z0-9_\-]{11})", url)
    if not video_id_match:
        return "yt_error:id_not_found"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
        text = " ".join(piece["text"] for piece in transcript)
        return text[:chars]
    except Exception as e:
        return f"yt_error:{e}"
    
# ==================================================================================
tools = [
    web_search,
    wikipedia_search,
    arxiv_search,
    add, subtract, multiply, divide, power, modulus, square_root,
    pdf_reader,
    spreadsheet_reader,
    image_ocr,
    code_file_interpreter, 
    audio_transcriber,
    youtube_transcript,
    
]


def build_graph(provider: str = "google"):
    """Build the graph"""
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model=groq_model, temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
            model=huggingFace_model,
            # huggingfacehub_api_token=os.environ["HF_TOKEN"],
            temperature=0,
          )
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        """Assistant node"""
        with open('system_prompt.txt', 'r') as f:
            system_prompt = f.read()
        sys_msg = SystemMessage(content=system_prompt)

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