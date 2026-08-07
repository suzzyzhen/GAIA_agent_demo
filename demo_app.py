"""
demo_app.py
Portfolio demo for the GAIA agent.

Usage:
    python demo_app.py
"""

import os
import mimetypes
import shutil
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from huggingface_hub import hf_hub_download
from agent import build_graph, extract_final_answer

AUDIO_TASK_FILENAME = "2023/validation/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"
PDF_TASK_FILENAME = "2023/validation/366e2f2b-8632-4ef2-81eb-bc3877489217.pdf"
IMAGE_TASK_FILENAME = "2023/test/7245af7c-404e-4d60-9ef4-94ed301e5315.jpg"


def download_gaia_file(filename: str, local_name: str) -> str:
    """Download a GAIA file and copy it into the local working directory
    so Gradio's file-serving checks accept it (see: InvalidPathError)."""
    downloaded_path = hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        filename=filename,
        token=os.environ.get("HF_TOKEN"),
    )
    os.makedirs("multimodal_data", exist_ok=True)
    local_path = os.path.join("multimodal_data", local_name)
    shutil.copy(downloaded_path, local_path)
    return local_path


DOWNLOADED_FILE_PATHS = {
    "audio": download_gaia_file(AUDIO_TASK_FILENAME, "example_audio.mp3"),
    "pdf": download_gaia_file(PDF_TASK_FILENAME, "example_document.pdf"),
    "image": download_gaia_file(IMAGE_TASK_FILENAME, "example_image.png"),
}


EXAMPLES = [
    ["What's the population of France divided by 2?", None],
    ["On the BBC Earth YouTube video of the Top 5 Silliest Animal Moments, what species of bird is featured?", None],
    [
        """Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3. In your response, please only list the ingredients, not any measurements. So if the recipe calls for "a pinch of salt" or "two cups of ripe strawberries" the ingredients on the list would be "salt" and "ripe strawberries". Please format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients.""",
        DOWNLOADED_FILE_PATHS["audio"],
    ],
    [
        "The attached file lists accommodations in the resort town of Seahorse Island. Based on the information in this file, \
        which seems like the better available place to stay for a family that enjoys swimming and wants a full house?",
        DOWNLOADED_FILE_PATHS["pdf"],
    ],
    [
        "The paint sample in the upper center of the attached image has a punny name. What word is the sample's name meant to sound like?",
        DOWNLOADED_FILE_PATHS["image"],
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
    "run_python_code": "Code executor",
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


def preview_attachment(file_path: str | None):
    if not file_path:
        return (
            gr.update(visible=False, value=None),  # image
            gr.update(visible=False, value=None),  # audio
            gr.update(visible=False, value=None),  # pdf
            gr.update(visible=False, value=None),  # general files
        )

    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or ""
    is_image = mime.startswith("image/")
    is_audio = mime.startswith("audio/")
    is_pdf = mime == "application/pdf"

    return (
        gr.update(visible=is_image, value=file_path if is_image else None),
        gr.update(visible=is_audio, value=file_path if is_audio else None),
        gr.update(visible=is_pdf, value=file_path if is_pdf else None),
        gr.update(visible=not (is_image or is_audio or is_pdf), value=file_path if not (is_image or is_audio or is_pdf) else None),
        gr.update(visible=True),
    )

# ================================== Format ==================================

CUSTOM_CSS = """
.gaia-card {
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06) !important;
    padding: 18px !important;
}
#final-answer-box textarea {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    border: 2px solid #6366f1 !important;
    border-radius: 12px !important;
    background: #eef2ff !important;
}
#header-title { text-align: center; margin-bottom: 0.1rem; }
#header-sub { text-align: center; color: #6b7280; margin-bottom: 0.5rem; }
"""

# ================================== Gradio ==================================
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=CUSTOM_CSS,
    title="GAIA agent demo",
) as demo:
    gr.Markdown("# GAIA agent demo", elem_id="header-title")
    gr.Markdown(
        "A LangGraph tool-calling agent that can search the web, read files, "
        "run calculations, transcribe audio, and more. "
        "[View the code on GitHub](https://github.com/suzzyzhen/GAIA_agent_demo)",
        elem_id="header-sub",
    )

    with gr.Group(elem_classes="gaia-card"):
        gr.Markdown("### Try an example")
        question_box = gr.Textbox(
            label="Question",
            placeholder="Ask the agent a question, or pick an example below...",
            lines=2,
        )
        file_box = gr.File(label="Attach a file (optional)", type="filepath")
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_box, file_box],
            label=None,
        )
        submit_button = gr.Button("Ask agent", variant="primary", size="lg")


    with gr.Group(elem_classes="gaia-card", visible=False) as preview_wrapper:
        gr.Markdown("### 📎 Attachment preview")
        preview_image = gr.Image(label="Image attachment", visible=False, interactive=False)
        preview_audio = gr.Audio(label="Audio attachment", visible=False, interactive=False)
        preview_file = gr.File(label="File attachment", visible=False, interactive=False)


    with gr.Accordion("🔍 Agent trace", open=True, elem_classes="gaia-card"):
        trace_output = gr.Markdown()

    with gr.Group(elem_classes="gaia-card"):
        gr.Markdown("### Final answer")
        answer_output = gr.Textbox(
            label=None, show_label=False, interactive=False, elem_id="final-answer-box"
        )

    file_box.change(
        fn=preview_attachment,
        inputs=file_box,
        outputs=[preview_image, preview_audio, preview_file, preview_wrapper],
    )

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