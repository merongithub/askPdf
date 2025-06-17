"""
PDF Question Answering Application using Gemini and Vertex AI

This Streamlit application allows users to:
1. Upload PDF documents
2. Process and chunk the documents
3. Store document embeddings in Vertex AI Vector Search
4. Ask questions about the document content
5. Get AI-generated answers using Gemini
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Disable Streamlit's file watcher for PyTorch
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from tempfile import NamedTemporaryFile
import torch
import sys
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Streamlit Cloud config override
if 'STREAMLIT_CLOUD' in os.environ:
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# === Configuration ===
CHUNK_SIZE = 50
PROJECT_ID = st.secrets["GCP_PROJECT_ID"]
LOCATION = st.secrets["GCP_LOCATION"]
INDEX_ID = st.secrets["VERTEX_AI_INDEX_ID"]
ENDPOINT_ID = st.secrets["VERTEX_AI_ENDPOINT_ID"]
DEPLOYED_INDEX_ID = "pdf_search_stream_deployed_index"

# Initialize Google Cloud credentials from secrets
credentials_dict = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
    "private_key": st.secrets["gcp_service_account"]["private_key"],
    "client_email": st.secrets["gcp_service_account"]["client_email"],
    "client_id": st.secrets["gcp_service_account"]["client_id"],
    "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
    "token_uri": st.secrets["gcp_service_account"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
}

# Create a temporary credentials file
with NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
    json.dump(credentials_dict, tmp)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

# Initialize AI models and services
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# === Load sentence embedding model ===
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Force CPU usage and disable JIT
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return embedder
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        st.stop()

# === Init models ===
try:
    embedder = load_models()
    chat_model = genai.GenerativeModel("models/gemini-1.5-pro")
    index = MatchingEngineIndex(index_name=INDEX_ID)
    endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_ID)
except Exception as e:
    st.error(f"Failed to initialize models or Vertex AI: {str(e)}")
    st.stop()

# === Chunk PDF ===
def read_and_chunk_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = "\n".join([page.get_text() for page in doc])
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

# === Streamlit App UI ===
st.title("🔍 Ask your PDF using Gemini + Vertex AI")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("📚 Processing PDF...")
    chunks = read_and_chunk_pdf(tmp_path)
    embeddings = embedder.encode(chunks).tolist()
    embeddings_array = np.array(embeddings)

    # === Upload chunks to Vertex AI ===
    try:
        index.upsert_datapoints(
            datapoints=[
                {
                    "datapoint_id": f"chunk-{i}",
                    "feature_vector": embedding.tolist()
                }
                for i, embedding in enumerate(embeddings_array)
            ]
        )
        #Sst.success(f"✅ {len(chunks)} chunks uploaded to Vertex AI.")
    except Exception as e:
        st.error(f"Failed to store chunks in Vertex AI: {str(e)}")
        st.stop()

    # === Ask question ===
    query = st.text_input("Ask a question about the PDF:")
    if query:
        st.info(f"🔍 Finding answers for: {query}")
        try:
            # Generate query embedding
            query_embedding = embedder.encode([query])[0]
            
            # Find nearest neighbors
            #st.info("Searching for relevant content...")
            results = endpoint.find_neighbors(
                deployed_index_id=DEPLOYED_INDEX_ID,
                queries=[query_embedding],
                num_neighbors=min(5, len(chunks))
            )
            
            if not results or len(results) == 0:
                st.warning("No relevant content found. Please try a different question.")
                st.stop()
                
            # Debug information
            #st.write(f"Found {len(results[0])} relevant chunks")
            
            # Get chunk text using datapoint IDs
            context_chunks = []
            for neighbor in results[0]:
                try:
                    chunk_id = neighbor.id
                    if chunk_id.startswith('chunk-'):
                        chunk_index = int(chunk_id.split('-')[1])
                        if 0 <= chunk_index < len(chunks):
                            context_chunks.append(chunks[chunk_index])
                        else:
                            logging.warning(f"Invalid chunk index: {chunk_index}")
                    else:
                        logging.warning(f"Unexpected chunk ID format: {chunk_id}")
                except Exception as e:
                    logging.warning(f"Error processing chunk {chunk_id}: {str(e)}")
            
            if not context_chunks:
                st.warning("No valid content chunks found. Please try a different question.")
                st.stop()
                
            context = "\n".join(context_chunks)
            
            # Generate answer
            #st.info("Generating answer...")
            prompt = f"""Use the following context to answer the question. If the information is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
            response = chat_model.generate_content(prompt)

            st.markdown("### 📖 Answer")
            st.write(response.text)

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.error("Please try again with a different question or re-upload the PDF.")

# Clean up temporary credentials file
if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
    try:
        os.unlink(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    except:
        pass
