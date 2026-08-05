"""
demo_app.py
Portfolio demo for the GAIA agent. 

Usage:
    python demo_app.py
"""

import os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from huggingface_hub import hf_hub_download
from agent import build_graph, extract_final_answer

SAMPLE_AUDIO_PATH = hf_hub_download(
    repo_id="gaia-benchmark/GAIA",
    repo_type="dataset",
    filename="2023/validation/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3",
)

EXAMPLES = [
    ["What's the population of France divided by 2?", None],
    ["Find the most recent arXiv paper on transformer efficiency and summarize its key finding.", None],
    [
        "Transcribe this audio clip and tell me what it says.",
        SAMPLE_AUDIO_PATH,
    ],
]

TOOL_LABELS = {
    "web_search": "Web search",
    "wikipedia_search": "Wikipedia search",
    "arxiv_search": "arXiv search",
    "pdf_reader": "PDF reader",
    "spreadsheet_reader": "Spreadsheet reader",
    "image_ocr": "Image OCR",
    "read_code_file": "Code reader",
    "execute_python_file": "Code executor",
    "audio_transcriber": "Audio transcriber",
    "youtube_transcript": "YouTube transcript",
}


def build_user_content(question: str, file_path: str | None) -> str:
    if file_path:
        return f"{question}\n\nAttached file path: {file_path}"
    return question


def run_agent(question: str, file_path: str | None, provider: str = "google"):
    """Run the agent, yielding (trace_markdown, final_answer) as the graph
    progresses so the UI updates live rather than waiting for the full run."""
    if not question or not question.strip():
        yield "", "Enter a question above to get started."
        return

    graph = build_graph(provider=provider)
    user_content = build_user_content(question, file_path)
    messages = [HumanMessage(content=user_content)]

    trace_lines = []
    final_answer = ""

    for step in graph.stream(
        {"messages": messages},
        config={"recursion_limit": 50},
        stream_mode="values",
    ):
        last_message = step["messages"][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for call in last_message.tool_calls:
                label = TOOL_LABELS.get(call["name"], call["name"])
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                trace_lines.append(f"**{label}** called with `{args}`")

        elif isinstance(last_message, ToolMessage):
            snippet = str(last_message.content).strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            trace_lines.append(f"> {snippet}")

        elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
            final_answer = extract_final_answer(last_message.content)

        trace_markdown = "\n\n".join(trace_lines) if trace_lines else "_No tool calls yet._"
        yield trace_markdown, final_answer


with gr.Blocks(title="GAIA agent demo") as demo:
    gr.Markdown(
        """
        # GAIA agent demo

        A LangGraph tool-calling agent that can search the web, read files,
        run calculations, transcribe audio, and more. Ask it a question
        below, or try one of the examples -- including an audio
        transcription example.

        [View the code on GitHub](https://github.com/suzzyzhen/GAIA_agent_demo)
        """
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Question",
            placeholder="Ask the agent a question",
            scale=4,
        )
        file_box = gr.File(
            label="Attach a file (optional)",
            type="filepath",
            scale=2,
        )

    submit_button = gr.Button("Ask agent", variant="primary")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_box, file_box],
        label="Try an example",
    )

    with gr.Accordion("Agent trace", open=False):
        trace_output = gr.Markdown()

    answer_output = gr.Textbox(label="Final answer", interactive=False)

    submit_button.click(
        fn=run_agent,
        inputs=[question_box, file_box],
        outputs=[trace_output, answer_output],
    )
    question_box.submit(
        fn=run_agent,
        inputs=[question_box, file_box],
        outputs=[trace_output, answer_output],
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)