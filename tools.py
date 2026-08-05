"""
tools.py
Collection of tools available to the agent: calculators, search tools,
and file/media processing tools.
"""

import os
import re
import subprocess
from typing import Optional

import arxiv
import pandas as pd
import pytesseract
import PIL.Image
import pdfplumber
import wikipediaapi
import whisper
from pdf2image import convert_from_path
from ddgs import DDGS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv


load_dotenv()
# =========================Calculator Tools=========================
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


# =========================Search Tools=========================

_tavily_search = TavilySearch(max_results=3, api_key=os.environ.get("TAVILY_API_KEY"))

_wiki_client = wikipediaapi.Wikipedia(
    user_agent="MyGAIAAgent/1.0 Huggingface course project",
    language="en",
)


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
        responses = _tavily_search.invoke(query)

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


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia. REQUIRED: you must provide a non-empty 'query' string
    parameter containing the search term, e.g. query='Alan Turing'."""

    page = _wiki_client.page(query)
    if not page.exists():
        return f"No Wikipedia page found for '{query}'."
    return page.summary[:2000]


# =========================File / Media Tools=========================

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
def read_code_file(file_path: str) -> str:
    """Read and return the raw source code of a file at the given path,
    without executing it. Use this to inspect code before running it.
    Args:
        file_path: Absolute path to the file to read."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def run_python_code(file_path: str) -> str:
    """Execute a Python (.py) file and return its stdout/stderr output.
    Args:
        file_path: Absolute path to the .py file to execute."""
    if not file_path.endswith(".py"):
        return "Error: only .py files can be executed. Use read_code_file to inspect other file types."
    if not os.path.isfile(file_path):
        return f"Error: file not found at {file_path}"
    try:
        result = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if result.returncode != 0:
            return f"Execution failed (exit code {result.returncode})\nSTDOUT:\n{output}\n\nSTDERR:\n{error}"
        return output if output else "Code executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out after 30 seconds."
    except Exception as e:
        return f"Error executing file: {e}"


# Load once and reuse across calls instead of reloading per-transcription.
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


@tool
def audio_transcriber(file_path: str) -> str:
    """Transcribe an audio file (mp3, wav, m4a, etc.) to text.
    Use this for any question that references an audio recording,
    lecture, voicemail, or similar attached audio file."""
    model = _get_whisper_model()
    result = model.transcribe(file_path)
    return result["text"].strip()


@tool
def youtube_transcript(url: str, chars: int = 10_000) -> str:
    """Fetch full YouTube transcript (first *chars* characters)."""
    video_id_match = re.search(r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_\-]{11})", url)
    if not video_id_match:
        return "yt_error:id_not_found"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
        text = " ".join(piece["text"] for piece in transcript)
        return text[:chars]
    except Exception as e:
        return f"yt_error:{e}"


# =========================Tool registry=========================

tools = [
    web_search,
    wikipedia_search,
    arxiv_search,
    add, subtract, multiply, divide, power, modulus, square_root,
    pdf_reader,
    spreadsheet_reader,
    image_ocr,
    read_code_file,
    run_python_code,
    audio_transcriber,
    youtube_transcript,
]