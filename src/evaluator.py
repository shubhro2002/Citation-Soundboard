from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class LineCategory(str, Enum):
    VERBATIM_FACT = "VERBATIM_FACT"
    INFERENCE = "INFERENCE"
    OPINION = "OPINION"

class EvaluatedLine(BaseModel):
    speaker: str = Field(description="The host speaking (e.g., 'Host A' or 'Host B')")
    text: str = Field(description="The actual spoken text of the script line")
    category: LineCategory = Field(description="Classification of the text groundedness")
    page_citation: Optional[int] = Field(
        description="If VERBATIM_FACT, extract the integer page number from the [Page X] tags in the source context. Else, output null."
    )
    reasoning: str = Field(description="Brief, one-sentence explanation of why this category was chosen")

class PodcastScript(BaseModel):
    lines: List[EvaluatedLine] = Field(description="The sequential list of evaluated podcast lines")

def get_evaluator_chain():
    """
    Builds a LangChain pipeline that forces the local Ollama model to output
    data strictly matching the PodcastScript Pydantic schema.
    """
    # Using llama3.2 with temperature 0 for deterministic, analytical results
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # Bind the Pydantic model to force structured JSON output
    structured_llm = llm.with_structured_output(PodcastScript)
    
    # System prompt directing the evaluator's behavior
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI self-evaluator enforcing strict groundedness in a podcast script. 
                      You will be provided with SOURCE CONTEXT containing page tags like '[Page 1]: text...', and a DRAFT SCRIPT.

                      Analyze every line of the script and classify it:
                      - VERBATIM_FACT: Hard data, metrics, or facts explicitly stated in the source.
                      - INFERENCE: Logical assumptions or connections based on the source.
                      - OPINION: Host banter, questions, or subjective commentary.

                      CRITICAL CITATION RULE:
                      For every VERBATIM_FACT line, you MUST look at the '[Page X]' tag in the SOURCE CONTEXT where that fact appears, and set 'page_citation' to that integer (e.g., 1 or 2). For INFERENCE or OPINION, set 'page_citation' to null."""),
        ("human", "SOURCE CONTEXT:\n{context}\n\nDRAFT SCRIPT:\n{script}")
    ])
    
    return prompt | structured_llm

if __name__ == "__main__":
    print("Evaluator schemas and chain initialized successfully.")