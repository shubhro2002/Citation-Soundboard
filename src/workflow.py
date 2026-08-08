import os
import operator
from pydantic import BaseModel, Field
from typing import TypedDict, cast, List, Annotated
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
    outline: List[str]               # Holds the generated subtopics
    current_index: int               # Tracks our position in the loop
    search_query: NotRequired[str]
    context: NotRequired[str]
    draft: NotRequired[str]
    full_script: Annotated[List[dict], operator.add]

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

class PodcastOutline(BaseModel):
    subtopics: List[str] = Field(
        description="A sequential list of 3 specific subtopics to discuss.",
        min_length=3, 
        max_length=3
    )

# 1.2 Node: Generate the Episode Outline
def outline_node(state: GraphState):
    print("\n=== GENERATING PODCAST OUTLINE ===")
    topic = state["topic"]
    
    llm = ChatOllama(model="llama3.2", temperature=0.6)
    structured_llm = llm.with_structured_output(PodcastOutline)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a podcast producer. Break the given topic into a logical, sequential 3-part outline. Make the subtopics highly specific so they can be used to search a document database."),
        ("human", "Topic: {topic}")
    ])
    
    response = cast(PodcastOutline, (prompt | structured_llm).invoke({"topic": topic}))
    
    print("  -> Generated Subtopics:")
    for i, sub in enumerate(response.subtopics):
        print(f"     {i+1}. {sub}")
        
    # Initialize the loop counter at 0, and the script as an empty list
    return {"outline": response.subtopics, "current_index": 0, "full_script": []}

def rewrite_node(state: GraphState):
    current_subtopic = state["outline"][state["current_index"]]
    print(f"\n--- TRANSFORMING QUERY (Subtopic {state['current_index'] + 1}) ---")
    print(f"  -> Target: {current_subtopic}")

    llm = ChatOllama(model="llama3.2", temperature=0.0)
    structured_llm = llm.with_structured_output(OptimizedSearch)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract only the core entities and keywords from the subtopic to create a dense vector search query. You are strictly forbidden from adding dates or years if they are not explicitly present."),
        ("human", "Subtopic: {topic}")
    ])  
    response = cast(OptimizedSearch, (prompt | structured_llm).invoke({"topic": current_subtopic}))
    optimized_query = response.query
    print(f"  -> Optimized Keywords: {optimized_query}")
    
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
    
    chain = get_evaluator_chain()
    eval_response = cast(PodcastScript, chain.invoke({"context": context, "script": draft}))
    evaluated_lines = [line.dict() for line in eval_response.lines]

    next_index = state["current_index"] + 1

    return {"full_script": evaluated_lines, "current_index": next_index}

def route_workflow(state: GraphState):
    # If we haven't processed all subtopics, loop back to the rewrite node
    if state["current_index"] < len(state["outline"]):
        return "rewrite"
    # Otherwise, finish the graph
    return END

def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("outline", outline_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("outline")
    workflow.add_edge("outline", "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "evaluate")
    workflow.add_conditional_edges(
        "evaluate", 
        route_workflow, 
        {"rewrite": "rewrite", END: END}
    )

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
        if "full_script" in final_state and final_state["full_script"]:
            for line in final_state["full_script"]:
                print(f"[{line['category']}] {line['speaker']} (Page {line['page_citation']}): {line['text']}")
                print(f"  -> Reasoning: {line['reasoning']}\n")