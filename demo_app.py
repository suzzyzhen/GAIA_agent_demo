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
import shutil

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
    [   "The paint sample in the upper center of the attached image has a punny name. What word is the sample’s name meant to sound like?",
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

    answer_output = gr.Textbox(label="Final answer", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_box, file_box],
        label="Try an example",
    )

    with gr.Accordion("Agent trace", open=True):
        trace_output = gr.Markdown()


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