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
    graph = build_workflow()
    
    initial_state = cast(GraphState, {"topic": topic})
    result = graph.invoke(initial_state)
    
    # Access the accumulated master script
    final_script_data = result.get("full_script")
    
    if not final_script_data:
        print("Error: Workflow failed to generate a script.")
        return
        
    print("\n=== FULL GENERATED & EVALUATED SCRIPT ===")
    for line in final_script_data:
        # Safe dict lookup using the explicit new schema keys
        category = line.get('step_2_category', 'UNKNOWN')
        speaker = line.get('speaker', 'Unknown')
        page = line.get('step_3_page_citation', 'None')
        text = line.get('text', '')
        reasoning = line.get('step_1_explanation_sentence', 'None')

        print(f"[{category}] {speaker} (Page {page}): {text}")
        print(f"  -> Reasoning: {reasoning}")
        
    print("\n[3/3] Synthesizing audio and stitching soundboard...")
    generate_podcast_audio(final_script_data, "podcast_output.wav")

if __name__ == "__main__":
    # Example usage
    SAMPLE_PDF = "./data/thesis.pdf"
    TOPIC = "How did the FLAN-T5 780M and 3B models perform during the iterative self-correction loops?"
    OUTPUT_AUDIO = "podcast_output.wav"

    run_citation_soundboard(SAMPLE_PDF, TOPIC, OUTPUT_AUDIO)