import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from tempfile import NamedTemporaryFile
from config import GEMINI_API_KEY, COLLECTION_NAME

# Config
CHUNK_SIZE = 50
COLLECTION_NAME = "pdf_chunks"
genai.configure(api_key=GEMINI_API_KEY)

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chat_model = genai.GenerativeModel("models/gemini-1.5-pro")

# Initialize ChromaDB (persistent)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)

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
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

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
        results = collection.query(query_texts=[query], n_results=5)
        context = "\n".join(results['documents'][0])
        prompt = f"""Use the following context to answer the question. If the information is not in the context, say so.\n\nContext:{context}\n\nQuestion: {query}\n\nAnswer:"""
        response = chat_model.generate_content(prompt)
        st.markdown("### 📖 Answer")
        st.write(response.text)
