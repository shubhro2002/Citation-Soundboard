import os
import glob
import pdfplumber
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512])

def ingest_directory_to_index(data_dir: str = "./data", persist_dir: str = "./storage") -> VectorStoreIndex:
    """
    Scans a directory for ALL PDFs, extracts text page-by-page with metadata, 
    and builds a single, unified local VectorStoreIndex.
    """
    print(f"Scanning directory '{data_dir}' for PDF files...")
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}. Please add your research papers.")

    documents = []

    # Loop through every PDF found in the folder
    for pdf_path in pdf_files:
        print(f"  -> Processing {os.path.basename(pdf_path)}...")
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Inject both the page number AND the source file name
                    doc = Document(
                        text=text,
                        metadata={
                            "page_number": i + 1, 
                            "source": os.path.basename(pdf_path)
                        }
                    )
                    documents.append(doc)
    
    print(f"\nExtracted a total of {len(documents)} pages across {len(pdf_files)} documents.")
    print("Building unified Vector Index (this may take a moment)...")

    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    print("Building unified Vector Index on leaf nodes for pinpoint accuracy...")

    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Hierarchical Index successfully persisted to {persist_dir}")
    
    return index

if __name__ == "__main__":
    # Now you just point it at the folder, not a specific file!
    DATA_DIR = "./data" 
    
    if os.path.exists(DATA_DIR):
        index = ingest_directory_to_index(DATA_DIR)
        print("Batch ingestion complete!")
    else:
        print(f"Directory {DATA_DIR} not found. Please create it and add PDFs.")