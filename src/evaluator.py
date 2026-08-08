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
    speaker: str = Field(description="Exact speaker from the draft (Do not change)")
    text: str = Field(description="Exact text from the draft (CRITICAL: Do not change a single word).")

    # Force the model to think and verify first
    reasoning: str = Field(
        description="Write a full English sentence explaining if the text contains hard metrics from the source. Do NOT output the category name here."
    )

    # Now the model can accurately classify based on its own reasoning
    category: LineCategory = Field(
        description="Classification of the text groundedness. Must be one of: VERBATIM_FACT, INFERENCE, OPINION."
    )

    # Finally, extract the page number
    page_citation: Optional[int] = Field(
        description="If VERBATIM_FACT, extract strictly the number following the [Page X] tag at the start of the chunk. Else, output null."
    )

class PodcastScript(BaseModel):
    line_1: EvaluatedLine = Field(description="Evaluation of line 1")
    line_2: EvaluatedLine = Field(description="Evaluation of line 2")
    line_3: EvaluatedLine = Field(description="Evaluation of line 3")
    line_4: EvaluatedLine = Field(description="Evaluation of line 4")

    @property
    def lines(self):
        return [self.line_1, self.line_2, self.line_3, self.line_4]

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
- VERBATIM_FACT: Hard data, metrics, or facts explicitly stated in the source, EVEN IF they are wrapped in conversational framing (e.g., "The numbers show a 45% increase"). If the core data point is real, it is a fact.
- INFERENCE: Logical assumptions or connections based on the source.
- OPINION: Host banter, questions, or subjective commentary that contains no hard source data.

CRITICAL CITATION RULE:
For every VERBATIM_FACT line, you MUST look at the '[Page X]' tag in the SOURCE CONTEXT where that fact appears, and set 'page_citation' to that integer. For INFERENCE or OPINION, set 'page_citation' to null."""),
        ("human", "SOURCE CONTEXT:\n{context}\n\nDRAFT SCRIPT:\n{script}")
    ])
    
    return prompt | structured_llm

if __name__ == "__main__":
    print("Evaluator schemas and chain initialized successfully.")