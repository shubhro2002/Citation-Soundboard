# Agentic Podcast Studio

An autonomous, fully offline, privacy-first AI podcast generator.

This project takes a folder of PDF research papers and transforms them into a studio-quality, multi-speaker podcast. It uses a LangGraph Agentic Swarm to draft and self-correct the script, LlamaIndex Hierarchical Retrieval for deep contextual accuracy, and Kokoro TTS + Pydub for dynamic voice assignment and cinematic audio post-production.

## Key Features

- **Offline & Privacy-First**: Powered entirely by local models via Ollama. Your confidential research papers never leave your machine.

- **Hierarchical Parent-Child Retrieval**: Uses LlamaIndex's *AutoMergingRetriever*. It searches tiny granular sentences for keyword precision, but automatically merges them into massive parent paragraphs before feeding the LLM, drastically reducing hallucinations and metadata loss.

- **Agentic Self-Correction**: Features a built-in "Multi-Dimensional Semantic Integrity Critic." If the drafting LLM hallucinates a metric or breaks the JSON schema, the Critic catches it and forces a rewrite before the audio is generated.

- **Global Conversational Memory**: The Map-Reduce pipeline passes state between loops, ensuring smooth, natural transitions across different subtopics rather than sounding like disjointed facts.

- **Dynamic Audio Post-Production**:

  - **Auto-Casting**: Automatically detects emergent speaker names in the script and assigns them unique Kokoro voices on the fly.

  - **Natural Pacing**: Programmatically injects 600ms pauses between speakers for human-like breathing room.

  - **Audio Ducking**: Automatically loops a background music track (`bg_music.mp3`), ducking the volume down during speech and fading it up during transitions.

- **Streamlit Web Dashboard**: A sleek UI to upload PDFs, watch the LangGraph swarm execute in real-time, read the color-coded cited script, and listen to the final episode.

## System Architecture

```mermaid
graph TD
    subgraph Phase 1: Ingestion
        A[Research PDFs] --> B(pdfplumber)
        B --> C{Hierarchical Chunking}
        C -->|Child Sentences| D[(Vector Index)]
        C -->|Parent Paragraphs| E[(Docstore)]
    end

    subgraph Phase 2: LangGraph Orchestration
        F[User Topic] --> G[Outline Node]
        G --> H[Retrieve Node<br/>Auto-Merging]
        D -.->|Keyword Match| H
        E -.->|Context Merge| H
        H --> I[Draft Node<br/>Llama 3.2]
        I --> J{Evaluate Node<br/>Critic}
        J -->|Failed check: Rewrite| I
        J -->|Passed: Append| K[Global Memory]
        K -->|Next Subtopic| H
    end

    subgraph Phase 3: Audio Synthesis
        K -->|Final Script| L[Kokoro TTS]
        L --> M[Speech Segments]
        M --> N[Pydub Post-Production]
        O[bg_music.mp3] -.->|Volume Ducking| N
        N --> P([🎙️ podcast_output.wav])
    end
```
1. **Ingestion Phase (`src/ingestion.py`)**: pdfplumber extracts text and page numbers. `HierarchicalNodeParser` chunks the text into a family tree of 2048-token Parent paragraphs and 512-token Child sentences, stored locally.

2. **Orchestration Phase (`src/workflow.py`)**:

   - User inputs a topic.

  - The LLM writes an outline of 3 subtopics.

  - Loop: For each subtopic -> Retrieve Context -> Draft Script -> Critic Evaluates -> (Retry if failed) -> Save to Global Memory.

3. **Synthesis Phase (`src/audio.py`)**: The finalized JSON script is parsed. Kokoro TTS generates WAV files per line. Pydub stitches them together, applies pacing, and mixes the background music.

## Project Structure

```
.
├── app.py                  # Streamlit frontend dashboard
├── main.py                 # CLI execution script (Alternative to Streamlit)
├── bg_music.mp3            # (User added) Background audio track
├── src/
│   ├── workflow.py         # LangGraph nodes, state, and Auto-Merging Retrieval
│   ├── ingestion.py        # PDF parsing and Hierarchical Vector DB creation
│   └── audio.py            # Kokoro TTS generation & Pydub post-production
├── data/                   # Directory where uploaded PDFs are stored
└── storage/                # Auto-generated LlamaIndex database

```

## Installation & Setup

This project requires a mix of system-level audio dependencies (for TTS phonetics and audio mixing) and Python packages.

### 1. System Dependencies (OS Level)

You must install FFmpeg (for PyDub audio manipulation) and eSpeak-NG (the phonetic engine required by Kokoro TTS to sound out words).

- **macOS**:

```bash
brew install ffmpeg espeak-ng
```
- **Linux (Ubuntu/Debian)**:

```bash
sudo apt update
sudo apt install ffmpeg espeak-ng
```

- **Windows**

  - Install FFmpeg via Winget: `winget install ffmpeg`

  - Install eSpeak-NG via Winget: `winget install eSpeak-NG.eSpeak-NG` (Or download the installer from the eSpeak-NG GitHub releases).

### 2. Ollama Setup (Local LLMs)

Install Ollama for your operating system, then open your terminal and pull the required language and embedding models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Python Environment Setup

It is highly recommended to use a virtual environment (Python 3.10+) to avoid package conflicts.

```bash
# Create and activate a virtual environment
python -m venv podcast_env

# On Mac/Linux:
source podcast_env/bin/activate
# On Windows:
podcast_env\Scripts\activate
```
### 4. Install Python Packages

With your virtual environment activated, install the required libraries:

```bash
pip install -r requirements.txt
```
*(A note on Kokoro TTS: By using `from kokoro import KPipeline`, the package will automatically download the required Kokoro model weights and voice pack `.pt` files directly from HuggingFace on your very first run. This means your first podcast generation will take a little longer as it caches the voices, but will be completely offline for all subsequent runs)*.

## How to Use

### 1. Optional: Add Background Music

For the full studio experience, download any royalty-free lo-fi or ambient MP3 track, name it `bg_music.mp3`, and place it directly in the sub-directory (`src/`). The audio engine will automatically find it and apply dynamic volume ducking.

### 2. Launch the Studio

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

### 3. Generate a Podcast

1. Open the web interface (usually `http://localhost:8501`).

2. Use the sidebar to upload your research PDFs.

3. Click "💾 Save & Build Database". (This builds the Hierarchical Vector DB).

4. Enter your desired topic in the main text box (e.g., *"How do large language models perform during iterative self-correction?"*).

5. Click "🚀 Generate Podcast".

6. Watch the swarm stream its thought process live. Once finished, read the cited script and hit play on the audio widget!

## Future Upgrades (Cloud LLM Integration)

While this project is optimized for a fully local llama3.2 8B model, the cognitive capabilities of the Agentic Critic can be pushed to flawless accuracy by swapping the local LLM for a frontier model.

To upgrade the Evaluator Node to **GPT-5.6 Terra** or **Llama 3 70B**:

1. `pip install langchain-openai`

2. Update the `evaluate_node` in `src/workflow.py` to use ChatOpenAI pointing to the OpenRouter API.

*Built with LangGraph, LlamaIndex, and a lot of Agentic Engineering.*