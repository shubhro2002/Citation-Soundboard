import os
import shutil
import streamlit as st
from typing import cast

from src.ingestion import ingest_directory_to_index
from src.workflow import build_workflow, GraphState
from src.audio import generate_podcast_audio

DATA_DIR = "./data"
STORAGE_DIR = "./storage"
OUTPUT_AUDIO = "podcast_output.wav"

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Agentic Podcast Studio", page_icon="🎙️", layout="wide")

st.title("🎙️ Agentic Podcast Studio")
st.markdown("Upload research papers, type a topic, and let the LangGraph swarm generate a fully voiced, cited podcast.")

# ==========================================
# SIDEBAR: DATA MANAGEMENT
# ==========================================
with st.sidebar:
    st.header("📚 Knowledge Base")
    
    # 1. File Uploader
    uploaded_files = st.file_uploader("Upload Research Papers (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("💾 Save & Build Database"):
        if uploaded_files:
            with st.spinner("Saving files and building Vector Database..."):
                # Clear old data and storage to prevent ghost data
                if os.path.exists(DATA_DIR):
                    shutil.rmtree(DATA_DIR)
                os.makedirs(DATA_DIR, exist_ok=True)
                
                if os.path.exists(STORAGE_DIR):
                    shutil.rmtree(STORAGE_DIR)
                
                # Save new files
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Trigger batch ingestion
                ingest_directory_to_index(DATA_DIR)
            st.success("Database successfully built! Ready to generate.")
        else:
            st.warning("Please upload at least one PDF first.")

    st.divider()
    
    # Show currently loaded files
    st.subheader("Current Database")
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".pdf"):
                st.markdown(f"- 📄 `{f}`")
    else:
        st.markdown("*No papers loaded.*")

# ==========================================
# MAIN AREA: PODCAST GENERATION
# ==========================================
topic = st.text_input("🎯 What should the podcast be about?", placeholder="e.g., Can humans reliably detect misinformation?")

if st.button("🚀 Generate Podcast", type="primary"):
    if not topic:
        st.warning("Please enter a topic first.")
    elif not os.path.exists(STORAGE_DIR):
        st.error("No database found! Please upload a PDF and click 'Build Database' in the sidebar.")
    else:
        with st.status("🧠 Awakening the Agentic Swarm...", expanded=True) as status_box:
            
            st.write("🏗️ Building the LangGraph workflow...")
            graph = build_workflow()
            initial_state = cast(GraphState, {"topic": topic})
            
            # NEW: Create a master list to accumulate the script chunks
            final_script_data = []
            
            st.write("🔍 Searching Vector Database and Drafting Script (This may take a few minutes)...")
            
            for output in graph.stream(initial_state):
                for node_name, state_update in output.items():
                    st.write(f"🔄 **Node Complete:** `{node_name}`")
                    
                    # NEW: Every time the evaluator approves a segment, append it to our master list!
                    if "full_script" in state_update:
                        final_script_data.extend(state_update["full_script"])
            
            st.write("✅ Script Finalized! Synthesizing Audio...")
            
            # Now we use our manually accumulated final_script_data
            if len(final_script_data) > 0:
                generate_podcast_audio(final_script_data, OUTPUT_AUDIO)
                status_box.update(label="🎉 Podcast Generation Complete!", state="complete", expanded=False)
            else:
                status_box.update(label="❌ Generation Failed", state="error", expanded=True)
                st.stop()

        # ==========================================
        # RESULTS DISPLAY
        # ==========================================
        st.header("🎧 Listen to the Episode")
        st.audio(OUTPUT_AUDIO, format="audio/wav")
        
        st.header("📜 Generated Script & Citations")
        
        for line in final_script_data:
            category = line.get('step_2_category', 'UNKNOWN')
            speaker = line.get('speaker', 'Unknown')
            page = line.get('step_3_page_citation', 'None')
            text = line.get('text', '')
            reasoning = line.get('step_1_explanation_sentence', 'None')
            
            badge_color = "blue"
            if category == "VERBATIM_FACT": badge_color = "green"
            elif category == "INFERENCE": badge_color = "orange"
            elif category == "OPINION": badge_color = "violet"
            
            with st.chat_message("user" if "Host" in speaker else "assistant"):
                st.markdown(f":{badge_color}[**[{category}]**] **{speaker}** (Page {page})")
                st.markdown(f"> {text}")
                with st.expander("Show AI Reasoning"):
                    st.caption(f"**Critic's Logic:** {reasoning}")