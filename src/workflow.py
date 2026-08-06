import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from evaluator import get_evaluator_chain, PodcastScript

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

class GraphState(TypedDict):
    topic: str
    context: str
    draft: str
    final_script: PodcastScript

def retrieve_node(state: GraphState):
    print("--- RETRIEVING CONTEXT ---")
    topic = state["topic"]

    # Load the index from storage
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(topic)

    context_str = ""
    for node in response.source_nodes:
        page = node.metadata.get('page_number', 'Unknown')
        context_str += f"[Page {page}]: {node.text}\n\n"
        
    return {"context": context_str}

def draft_node(state: GraphState):
    print("--- DRAFTING SCRIPT ---")
    context = state["context"]

    llm =  ChatOllama(model="llama3.2", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a podcast writer. Write a short, engaging 4-line back-and-forth dialogue between 'Host A' and 'Host B' about the provided context. Include at least one specific metric or fact from the text."),
        ("human", "CONTEXT:\n{context}\n\nDRAFT SCRIPT:")
        ])

    chain = prompt | llm
    draft_response = chain.invoke({"context": context})

    return {"draft": draft_response}

def evaluate_node(state: GraphState):
    print("--- EVALUATING GROUNDEDNESS ---")
    context = state["context"]
    draft = state["draft"]
    
    eval_chain = get_evaluator_chain()
    structured_script = eval_chain.invoke({"context": context, "script": draft})
    
    return {"final_script": structured_script}

def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()

if __name__ == "__main__":
    if not os.path.exists("./storage"):
        print("Error: No ./storage folder found. Run ingestion.py first!")
    else:
        app = build_workflow()
        inputs = {"topic": "Acme Corp Q2 financial expenses and server farm acquisition"}
        print(f"Starting graph with topic: {inputs['topic']}")

        final_state = app.invoke(inputs)

        print("\n=== FINAL EVALUATED SCRIPT ===")
        if "final_script" in final_state and final_state["final_script"]:
            for line in final_state["final_script"].lines:
                print(f"[{line.category}] {line.speaker} (Page {line.page_citation}): {line.text}")
                print(f"  -> Reasoning: {line.reasoning}\n")