import os
from pydantic import BaseModel, Field
from typing import TypedDict, cast, List
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from .evaluator import get_evaluator_chain, PodcastScript

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

class GraphState(TypedDict):
    topic: str
    search_query: NotRequired[str]
    context: NotRequired[str]
    draft: NotRequired[str]
    final_script: NotRequired[PodcastScript]

class DraftLine(BaseModel):
    speaker: str = Field(description="Host A or Host B")
    text: str = Field(description="The spoken dialogue")

class DraftScript(BaseModel):
    line_1: DraftLine = Field(description="The first line of dialogue")
    line_2: DraftLine = Field(description="The second line of dialogue")
    line_3: DraftLine = Field(description="The third line of dialogue")
    line_4: DraftLine = Field(description="The fourth line of dialogue")

class OptimizedSearch(BaseModel):
    query: str = Field(description="The optimized keywords. CRITICAL: Do NOT add years or dates unless explicitly provided.")

def rewrite_node(state: GraphState):
    print("--- TRANSFORMING QUERY ---")
    topic = state["topic"]

    llm = ChatOllama(model="llama3.2", temperature=0.2)
    structured_llm = llm.with_structured_output(OptimizedSearch)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert search query optimizer for a vector database. Your goal is to take a user's conversational topic and rewrite it into a specific, keyword-rich search query. Output ONLY the optimized search query string, with no introductory text or quotes."),
        ("human", "USER TOPIC:\n{topic}\n\nSEARCH QUERY:")
    ])
    chain = prompt | structured_llm
    response = cast(OptimizedSearch, chain.invoke({"topic": topic}))
    
    optimized_query = response.query
    print(f"  -> Optimized Query: {optimized_query}")
    
    return {"search_query": optimized_query}

def retrieve_node(state: GraphState):
    print("--- RETRIEVING CONTEXT ---")
    query = state.get("search_query", state["topic"])  # Use search_query if available, else fallback to topic

    # Load the index from storage
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)

    context_str = ""
    for node in response.source_nodes:
        page = node.metadata.get('page_number', 'Unknown')
        context_str += f"[Page {page}]: {node.text}\n\n"

    # Debug
    print(f"\n--- DEBUG RETRIEVED CONTEXT ---\n{context_str}--------------------------------\n")
    return {"context": context_str}

def draft_node(state: GraphState):
    print("--- DRAFTING SCRIPT ---")
    context = state.get("context", "")

    llm =  ChatOllama(model="llama3.2", temperature=0.7)
    structured_llm = llm.with_structured_output(DraftScript)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a podcast writer. Write an engaging 4-line back-and-forth dialogue between 'Host A' and 'Host B' using the provided context. You must include at least one specific metric or fact. Do not repeat lines."),
        ("human", "CONTEXT:\n{context}\n\nDRAFT SCRIPT:")
    ])

    chain = prompt | structured_llm
    draft_response = cast(DraftScript, chain.invoke({"context": context}))

    draft_str = (
        f"{draft_response.line_1.speaker}: {draft_response.line_1.text}\n"
        f"{draft_response.line_2.speaker}: {draft_response.line_2.text}\n"
        f"{draft_response.line_3.speaker}: {draft_response.line_3.text}\n"
        f"{draft_response.line_4.speaker}: {draft_response.line_4.text}\n"
    )
        
    return {"draft": draft_str}

def evaluate_node(state: GraphState):
    print("--- EVALUATING GROUNDEDNESS ---")
    context = state.get("context", "")
    draft = state.get("draft", "")
    
    eval_chain = get_evaluator_chain()
    structured_script = eval_chain.invoke({"context": context, "script": draft})
    
    return {"final_script": structured_script}

def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()

if __name__ == "__main__":
    if not os.path.exists("./storage"):
        print("Error: No ./storage folder found. Run ingestion.py first!")
    else:
        app = build_workflow()
        inputs = cast(
            GraphState, 
            {"topic": "Acme Corp Q2 financial expenses and server farm acquisition"}
        )
        print(f"Starting graph with topic: {inputs['topic']}")

        final_state = app.invoke(inputs)

        print("\n=== FINAL EVALUATED SCRIPT ===")
        if "final_script" in final_state and final_state["final_script"]:
            for line in final_state["final_script"].lines:
                print(f"[{line.category}] {line.speaker} (Page {line.page_citation}): {line.text}")
                print(f"  -> Reasoning: {line.reasoning}\n")