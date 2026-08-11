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
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

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

    # State trackers for self-correction loops
    retry_count: int
    feedback: NotRequired[str]

    # Memory of the previous segment
    previous_context: NotRequired[str]

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

# Generate the Episode Outline
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
    return {"outline": response.subtopics, "current_index": 0, "full_script": [], "retry_count": 0, "previous_context": ""}

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
    current_subtopic = state["outline"][state["current_index"]]
    query = state.get("search_query", current_subtopic)
    
    print(f"\n--- RETRIEVING CONTEXT FOR: {query} ---")
    
    # Load the existing Index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    # Initialize Dense Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=5)
    
    # Initialize Sparse BM25 Retriever
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, 
        similarity_top_k=5
    )
    
    # Combine via Reciprocal Rank Fusion
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",  # type: ignore
        use_async=False,
    )
    
    # Execute Hybrid Search
    nodes = hybrid_retriever.retrieve(query)
    
    context_str = ""
    print("\n--- DEBUG RETRIEVED CONTEXT ---")
    for node in nodes:
        # Access the underlying TextNode metadata dictionary safely
        node_obj = node.node if hasattr(node, "node") else node
        metadata = getattr(node_obj, "metadata", {})
        
        # Check all standard LlamaIndex page number keys
        page_num = (
            metadata.get("page_label") or 
            metadata.get("page_number") or 
            metadata.get("page_num") or 
            "Unknown"
        )
        
        preview = node_obj.get_content()[:150].replace('\n', ' ')
        print(f"[Page {page_num}]: {preview}...\n")
        context_str += f"[Page {page_num}]: {node_obj.get_content()}\n\n"
        
    return {"context": context_str}

def draft_node(state: GraphState):
    print(f"--- DRAFTING SCRIPT (Attempt {state.get('retry_count', 0) + 1}) ---")
    context = state.get("context", "")
    feedback = state.get("feedback", "")
    previous_context = state.get("previous_context", "")
    
    llm = ChatOllama(model="llama3.2", temperature=0.7)
    structured_llm = llm.with_structured_output(DraftScript)
    
    # If we are in a correction loop, append the strict feedback
    system_prompt = "You are a podcast writer. Write an engaging 4-line back-and-forth dialogue using the provided context."
    if previous_context:
        system_prompt += (
            f"\n\nPREVIOUS SEGMENT:\n{previous_context}\n\n"
            "CRITICAL INSTRUCTION: The user just heard the PREVIOUS SEGMENT. The first line of your new draft MUST naturally transition "
            "from that previous conversation into the new SOURCE CONTEXT. Do not introduce yourselves again."
        )
    if feedback:
        print(f"  -> Applying Correction: {feedback}")
        system_prompt += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS DRAFT: {feedback}. You MUST fix this in your new draft."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "SOURCE CONTEXT:\n{context}")
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
    current_retries = state.get("retry_count", 0)
    MAX_RETRIES = 2  # Our T_max computational budget
    
    chain = get_evaluator_chain()
    eval_response = cast(PodcastScript, chain.invoke({"context": context, "script": draft}))
    evaluated_lines = [line.dict() for line in eval_response.lines]
    
    # Agentic Evaluation Logic: Count valid facts
    valid_facts = sum(
        1 for line in evaluated_lines 
        if line.get('step_2_category') == "VERBATIM_FACT" and line.get('step_3_page_citation') is not None
    )
    
    # If we fail the threshold AND have retries left, trigger a loop
    if valid_facts < 2 and current_retries < MAX_RETRIES:
        print(f"  -> CRITIC FAILED SCRIPT: Only {valid_facts} valid citations found. Triggering Loop.")
        feedback_msg = "Your previous draft lacked hard facts or page numbers. You must include at least two concrete metrics, numbers, or specific facts from the text."
        return {"feedback": feedback_msg, "retry_count": current_retries + 1}
        
    # If we pass, OR we hit T_max, we forcefully accept the output and move to the next subtopic
    if current_retries >= MAX_RETRIES:
        print("  -> T_MAX REACHED: Forcing premature halt to prevent infinite loop.")
    else:
        print("  -> CRITIC PASSED SCRIPT.")
        
    next_index = state["current_index"] + 1

    saved_context = "\n".join([f"{line['speaker']}: {line['text']}" for line in evaluated_lines])
    
    # Reset retry count and clear feedback for the next subtopic
    return {
        "full_script": evaluated_lines, 
        "current_index": next_index, 
        "retry_count": 0, 
        "feedback": "",
        "previous_context": saved_context
    }

def route_workflow(state: GraphState):
    # If the evaluator attached feedback and incremented the retry count, loop backward
    if state.get("feedback") and state.get("retry_count", 0) > 0:
        return "draft"
        
    # If we haven't processed all subtopics, loop forward
    if state["current_index"] < len(state["outline"]):
        return "rewrite"
        
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
        {"rewrite": "rewrite", "draft": "draft", END: END}
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