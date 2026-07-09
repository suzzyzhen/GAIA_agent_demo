from langchain_core.tools import tool
import os
import arxiv
import wikipediaapi
import pdfplumber
from pdf2image import convert_from_path
import pandas as pd
import pytesseract
from PIL import Image
import subprocess

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
    """Search Tavily for a query and return maximum 2 results.
    Args:
        query: The search query."""

    search = TavilySearch(max_results=3)
    responses = search.invoke(query)

    formatted_responses = "\n\n".join(
      f"""[{i}]
        Title: {doc.get("title", "")}
        URL: {doc.get("url", "")}
        Content: {doc.get("content", "")}
      """
      for i, doc in enumerate(responses["results"], start=1)
      )
    return {"web_results": formatted_responses}

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
    wiki_client = wikipediaapi.Wikipedia(
    user_agent="MyGAIAAgent/1.0 (myemail@example.com)",
    language="en"
)
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
def spreadsheet_reader(file_path: str, sheet_name: str = None) -> str:
    """Read a CSV or Excel file at the given local file path and return its
    contents as text. Optionally specify sheet_name for a specific Excel sheet;
    if omitted, all sheets are returned."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_string(index=False)

    if sheet_name:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df.to_string(index=False)
    else:
        sheets = pd.read_excel(file_path, sheet_name=None)
        parts = [f"Sheet: {name}\n{df.to_string(index=False)}" for name, df in sheets.items()]
        return "\n\n---\n\n".join(parts)
    
@tool
def image_ocr(file_path: str) -> str:
    """Extract any visible text from an image file using OCR.
    Best for screenshots, scanned documents, charts with labels, or text-heavy images."""
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text.strip() if text.strip() else "No text found in image."


@tool
def code_file_interpreter(file_path: str, mode: str = "execute") -> str:
    """Read or execute a code file at the given local file path.
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