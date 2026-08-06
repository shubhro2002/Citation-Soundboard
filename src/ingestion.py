import os
import pdfplumber
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def ingest_pdf_to_index(pdf_path: str, persist_dir: str = "./storage") -> VectorStoreIndex:
    """
    Reads a PDF using pdfplumber, extracts text page-by-page with metadata, 
    and builds a local VectorStoreIndex.
    """
    print(f"Loading PDF from {pdf_path} using pdfplumber...")
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Inject the page number into the metadata
                doc = Document(
                    text=text,
                    metadata={"page_number": i + 1, "source": os.path.basename(pdf_path)}
                )
                documents.append(doc)
    
    print(f"Extracted {len(documents)} pages. Building Vector Index...")

    # Build Index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index successfully persisted to {persist_dir}")
    
    return index

if __name__ == "__main__":
    SAMPLE_PDF = "./data/Sample_Data.pdf" 
    
    if os.path.exists(SAMPLE_PDF):
        index = ingest_pdf_to_index(SAMPLE_PDF)
        print("Ingestion test complete!")
    else:
        print(f"Please place a PDF at {SAMPLE_PDF} to test the ingestion.")