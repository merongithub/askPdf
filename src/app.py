import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from tempfile import NamedTemporaryFile
import torch
import os

# Config
CHUNK_SIZE = 50
COLLECTION_NAME = "pdf_chunks"
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return embedder
    except Exception as e:
        st.error(f"""
        ⚠️ Model Loading Error ⚠️
        
        Failed to load the sentence transformer model: {str(e)}
        
        Please try reinstalling the dependencies:
        ```bash
        pip uninstall -y sentence-transformers torch
        pip install -r requirements.txt
        ```
        """)
        st.stop()

# Initialize models
embedder = load_models()
chat_model = genai.GenerativeModel("models/gemini-1.5-pro")

# Initialize ChromaDB (in-memory)
try:
    import chromadb
    client = chromadb.PersistentClient(path="/tmp/chroma")  # Use a temporary directory
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    st.error(f"""
    ⚠️ ChromaDB Initialization Error ⚠️
    
    Failed to initialize ChromaDB: {str(e)}
    
    Please ensure all dependencies are correctly installed:
    ```bash
    pip install -r requirements.txt
    ```
    """)
    st.stop()

# Utility: Read and chunk PDF
def read_and_chunk_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = "\n".join([page.get_text() for page in doc])
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

# UI
st.title("🔍 Ask your PDF using Gemini + ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("📚 Processing PDF...")
    chunks = read_and_chunk_pdf(tmp_path)
    embeddings = embedder.encode(chunks).tolist()

    # Clear existing and store new chunks
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except:
        pass  # Collection might be empty

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))]
    )
    st.success(f"✅ {len(chunks)} chunks stored.")

    # Ask a question
    query = st.text_input("Ask a question about the PDF:")
    if query:
        st.info(f"🔍 Processing question: {query}")
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(5, len(chunks))
            )
            context = "\n".join(results['documents'][0])
            prompt = f"""Use the following context to answer the question. If the information is not in the context, say so.\n\nContext:{context}\n\nQuestion: {query}\n\nAnswer:"""
            response = chat_model.generate_content(prompt)
            st.markdown("### 📖 Answer")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
