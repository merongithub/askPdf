import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ["STREAMLIT_WATCH_TORCH"] = "false"
import streamlit as st
import fitz  # PyMuPDF
import torch
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from huggingface_hub import login
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Print available secrets
logger.info("Available secrets keys: %s", list(st.secrets.keys()))

# Optional: set CPU usage explicitly for Torch in cloud environments
if 'STREAMLIT_CLOUD' in os.environ:
    os.environ['PYTORCH_JIT'] = '0'

# Login to Hugging Face
try:
    login(token=st.secrets["HF_TOKEN"])
    logger.info("Successfully logged in to Hugging Face")
except Exception as e:
    logger.error("Failed to login to Hugging Face: %s", str(e))
    raise

# Constants
CHUNK_SIZE = 50
COLLECTION_NAME = "pdf_chunks"
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load embedding model
@st.cache_resource(show_spinner=False)
def load_models():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Initialize models
try:
    embedder = load_models()
    chat_model = genai.GenerativeModel("models/gemini-1.5-pro")
except Exception as e:
    st.error(f"❌ Failed to load models: {str(e)}")
    st.stop()

# Initialize ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    import shutil
    import os

    # Set up database directory
    chroma_path = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(chroma_path, exist_ok=True)
    print(f"Using ChromaDB at {chroma_path}")

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Create a new collection with explicit settings
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Using existing collection")
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None  # We'll handle embeddings separately
        )
        print("Created new collection")
except Exception as e:
    st.error(f"❌ ChromaDB Error: {str(e)}")
    st.stop()

# Chunking utility
def read_and_chunk_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

# Streamlit UI
st.title("📄 Ask Your PDF — RAG with Gemini + ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.info("🔍 Processing PDF...")
    chunks = read_and_chunk_pdf(pdf_path)
    embeddings = embedder.encode(chunks).tolist()

    # Clear previous entries
    try:
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass

    # Store new chunks
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))]
    )
    st.success(f"✅ {len(chunks)} chunks added.")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        st.info(f"🧠 Querying: {query}")
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(5, len(chunks))
            )
            context = "\n".join(results["documents"][0])
            prompt = f"""Use the context below to answer the question. If context lacks the info, say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
            response = chat_model.generate_content(prompt)
            st.markdown("### 📖 Answer")
            st.write(response.text)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
