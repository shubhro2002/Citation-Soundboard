import os
import sys
from typing import cast

from src.ingestion import ingest_pdf_to_index
from src.workflow import build_workflow, GraphState
from src.audio import generate_podcast_audio

def run_citation_soundboard(pdf_path: str, topic: str, output_audio: str = "podcast_output.wav"):
    print("=" * 50)
    print("       CITATION SOUNDBOARD: LOCAL PIPELINE       ")
    print("=" * 50)

    # Check if vector store index exists; if not, ingest the PDF
    if not os.path.exists("./storage"):
        print(f"\n[1/3] No existing storage found. Ingesting: {pdf_path}...")
        if not os.path.exists(pdf_path):
            print(f"Error: Target PDF '{pdf_path}' does not exist.")
            sys.exit(1)
        ingest_pdf_to_index(pdf_path)
    else:
        print("\n[1/3] Found existing storage index in ./storage. Skipping re-ingestion.")

    # Build and invoke the LangGraph workflow
    print("\n[2/3] Building and executing the LangGraph workflow...")
    app = build_workflow()

    inputs = cast(GraphState, {"topic": topic})
    final_state = app.invoke(inputs)

    # Process output and generate audio
    if "final_script" in final_state and final_state["final_script"]:
        evaluated_script = final_state["final_script"]
        
        print("\n=== GENERATED & EVALUATED SCRIPT ===")
        for line in evaluated_script.lines:
            print(f"[{line.category.value}] {line.speaker} (Page {line.page_citation}): {line.text}")
            print(f"  -> {line.reasoning}")
        
        print(f"\n[3/3] Synthesizing audio and stitching soundboard...")
        generate_podcast_audio(evaluated_script.lines, output_filename=output_audio)
        
        print("=" * 50)
        print(f"SUCCESS! Output audio ready at: {os.path.abspath(output_audio)}")
        print("=" * 50)
    else:
        print("Error: Graph execution failed to produce an evaluated script.")

if __name__ == "__main__":
    # Example usage
    SAMPLE_PDF = "./data/thesis.pdf"
    TOPIC = "How did the FLAN-T5 780M and 3B models perform during the iterative self-correction loops?"
    OUTPUT_AUDIO = "podcast_output.wav"

    run_citation_soundboard(SAMPLE_PDF, TOPIC, OUTPUT_AUDIO)