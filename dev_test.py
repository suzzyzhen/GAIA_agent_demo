from agent import build_graph
from langchain_core.messages import HumanMessage
import argparse
import textwrap
from typing import Any
import requests
import os

from huggingface_hub import hf_hub_download
from agent import build_graph, extract_final_answer

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# test_question = "What is the answer of 5 + 3?"

# def test_graph(question:str):
#     graph = build_graph(provider="huggingface") 
#     result = graph.invoke(
#         {"messages": [HumanMessage(content=question)]},
#         config={"recursion_limit": 50},
#     )
#     output = result["messages"][-1].content
#     print(output)
#     return output

def fetch_one_question(api_url: str = DEFAULT_API_URL) -> dict:
    """Fetch the question list and return just the first item."""
    response = requests.get(f"{api_url}/questions", timeout=15)
    response.raise_for_status()
    questions = response.json()
    if not questions:
        raise RuntimeError("No questions returned from the API.")
    return questions[0]
 
 
def resolve_file(file_name: str) -> str | None:
    """Download a GAIA task attachment and return its local path, if any."""
    if not file_name:
        return None
    return hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        filename=f"2023/validation/{file_name}",
        token=os.environ.get("HF_TOKEN"),
    )
 
 
def main():
    question_item = fetch_one_question()
    task_id = question_item.get("task_id")
    question_text = question_item.get("question")
    file_name = question_item.get("file_name", "")
 
    print(f"Task ID: {task_id}")
    print(f"Question: {question_text}")
    if file_name:
        print(f"Attached file: {file_name}")
 
    resolved_path = resolve_file(file_name)
    user_content = (
        f"{question_text}\n\nAttached file path: {resolved_path}"
        if resolved_path
        else question_text
    )
 
    # Swap provider here for quicker/cheaper local iteration, e.g. "google" or "groq"
    graph = build_graph(provider="huggingface")
 
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_content)]},
        config={"recursion_limit": 50},
    )
 
    raw_answer = result["messages"][-1].content
    final_answer = extract_final_answer(raw_answer)
 
    print("\n--- Raw agent output ---")
    print(raw_answer)
    print("\n--- Extracted final answer ---")
    print(final_answer)
 
 
if __name__ == "__main__":
    main()

    