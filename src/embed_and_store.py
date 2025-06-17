import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# Constants
CHUNK_SIZE = 50
COLLECTION_NAME = "pdf_chunks"

# Initialize models
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(COLLECTION_NAME)

def read_and_chunk_pdf(pdf_path: str) -> list:
    """Read PDF and split into chunks."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

def store_chunks(chunks: list):
    """Store chunks in ChromaDB."""
    # Generate embeddings
    embeddings = embedder.encode(chunks).tolist()
    
    # Clear existing collection by deleting it and recreating it
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    
    # Add new chunks
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))]
    )
    print(f"✅ {len(chunks)} chunks stored.")

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    chunks = read_and_chunk_pdf(pdf_path)
    store_chunks(chunks) 