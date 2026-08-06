---
title: GAIA Agent Demo (Huggingface Agents Course Project )
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: demo_app.py
pinned: false
---

# GAIA Agent Demo

A tool-calling AI agent built with [LangGraph](https://github.com/langchain-ai/langgraph), capable of web search, document reading, calculations, audio transcription, and more. Built as part of the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) [Hugingface AI Agents](https://huggingface.co/learn/agents-course/unit0/introduction)coursework, and adapted here into a live portfolio demo.

**[Try the live demo](https://huggingface.co/spaces/suzzy123/GAIA_agent_demo)**

## What this agent can do

The agent reasons step by step and calls tools as needed to answer a question, rather than relying purely on its own internal knowledge. Available tools:

| Tool | Purpose |
|---|---|
| `web_search` | General web search (DuckDuckGo, with Tavily fallback) |
| `wikipedia_search` | Look up a Wikipedia article summary |
| `arxiv_search` | Search academic papers on arXiv |
| `pdf_reader` | Extract text from a PDF (with OCR fallback for scanned documents) |
| `spreadsheet_reader` | Read CSV/Excel files |
| `image_ocr` | Extract text from an image |
| `audio_transcriber` | Transcribe audio files using Whisper |
| `youtube_transcript` | Fetch a YouTube video's transcript |
| `read_code_file` / `execute_python_file` | Inspect or run a Python file |
| `add`, `subtract`, `multiply`, `divide`, `power`, `modulus`, `square_root` | Arithmetic |

## Project structure
```
.
├── agent.py          # LangGraph agent: LLM setup, system prompt, graph construction
├── tools.py           # All tool definitions used by the agent
├── demo_app.py         # Portfolio demo (Gradio UI, no login, example questions)
├── app.py             # GAIA benchmark evaluation harness (HF login + scoring submission)
├── system_prompt.txt   # System prompt guiding the agent's behavior
├── requirements.txt     # Python dependencies
├── packages.txt        # System-level (apt) dependencies for the HF Space container
└── .github/workflows/    # GitHub Action that syncs this repo to the HF Space
```


## Architecture

- **`agent.py`** builds a LangGraph graph: a single LLM node bound to all tools, looping between the LLM and a `ToolNode` until the model responds without further tool calls. Supports three LLM providers (Google Gemini, Groq, Hugging Face) via a `provider` argument.
- **`tools.py`** defines each tool as a `@tool`-decorated function and exports a single `tools` list consumed by `agent.py`.
- **`demo_app.py`** is a standalone Gradio app for public demoing: a question box, example prompts (including file-based examples for transcription, PDF reading, and OCR), a live tool-call trace panel, and the final answer — no login or scoring involved.
- **`app.py`** is the original GAIA coursework harness: requires Hugging Face login, fetches the official question set, runs the agent on each question, and submits answers for scoring.

![Architecture](architecture.png)

## Running locally

**1. Clone and set up a virtual environment**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Install system dependencies**

A few tools shell out to system binaries not covered by `pip`:
```bash
# macOS
brew install ffmpeg tesseract poppler

# Debian/Ubuntu
sudo apt-get install ffmpeg tesseract-ocr poppler-utils
```

**3. Set environment variables**

Depending on which LLM provider(s) and tools you use:
```bash
export GOOGLE_API_KEY=...      # provider="google"
export GROQ_API_KEY=...        # provider="groq"
export HF_TOKEN=...            # provider="huggingface", and for gated dataset downloads
export TAVILY_API_KEY=...      # web_search fallback
```

**4. Run the demo**
```bash
python demo_app.py
```
Open the printed local URL (typically `http://127.0.0.1:7860`) in your browser.

## Deployment

This repo is the source of truth on GitHub. A GitHub Action (`.github/workflows/`) automatically syncs the latest `main` branch to the Hugging Face Space on every push, so the Space always mirrors this repo — no manual copying required.

The Space's hardware, SDK, and entrypoint (`app_file`) are configured via the YAML frontmatter at the top of this file.

## Notes

- Sample files used in the demo's example questions are downloaded at runtime from the GAIA dataset (not committed to this repo) to avoid storing binaries in git history.
- Source of multimodal files in the examples
[Audio](https://huggingface.co/datasets/gaia-benchmark/GAIA/blob/main/2023/validation/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3)
[Image](https://huggingface.co/datasets/gaia-benchmark/GAIA/blob/main/2023/validation/5b2a14e8-6e59-479c-80e3-4696e8980152.jpg)
[PDF](https://huggingface.co/datasets/gaia-benchmark/GAIA/blob/main/2023/validation/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf)


- `demo_app.py` (the public demo) and `app.py` (the GAIA scoring harness) and  are kept separate so the portfolio-facing demo never requires visitors to log in or triggers a benchmark submission.


