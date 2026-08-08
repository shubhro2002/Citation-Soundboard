from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class EvaluatedLine(BaseModel):
    speaker: str = Field(description="Exact speaker from the draft (Do not change)")
    text: str = Field(description="Exact text from the draft (CRITICAL: Do not change a single word).")

    # Force the model to think and verify first
    step_1_explanation_sentence: str = Field(
        description="Write a full English sentence explaining your thought process. Example: 'The text contains the metric 0.186'."
    )

    # Now the model can accurately classify based on its own reasoning
    step_2_category: Literal["VERBATIM_FACT", "INFERENCE", "OPINION"] = Field(
        description="Classify the text based on step 1. Must be VERBATIM_FACT if real metrics are present."
    )

    # Finally, extract the page number
    step_3_page_citation: Optional[int] = Field(
        description="If VERBATIM_FACT, extract strictly the number following the [Page X] tag. Else, output null."
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

Classify exactly 4 lines of the script.
- VERBATIM_FACT: Hard data, metrics, or facts explicitly stated in the source.
- INFERENCE: Logical assumptions or connections based on the source.
- OPINION: Host banter, questions, or subjective commentary.

CRITICAL RULES:
1. For 'step_1_explanation_sentence', write ONLY a normal English sentence. Do NOT use the words VERBATIM_FACT, INFERENCE, or OPINION here.
2. For 'step_3_page_citation', you MUST extract the integer from the '[Page X]' tag at the very top of the chunk. Do NOT use chapter or section numbers.

=== EXAMPLE OF PERFECT OUTPUT ===
"step_1_explanation_sentence": "The text explicitly states the 60M model dropped by -0.025.",
"step_2_category": "VERBATIM_FACT",
"step_3_page_citation": 29
================================="""),
        ("human", "SOURCE CONTEXT:\n{context}\n\nDRAFT SCRIPT:\n{script}")
    ])
    
    return prompt | structured_llm

if __name__ == "__main__":
    print("Evaluator schemas and chain initialized successfully.")