import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import requests
from tempfile import NamedTemporaryFile
import os

# Config
API_URL = "http://localhost:8000"  # Change this to your FastAPI server URL

# UI
st.title("🔍 Ask your PDF using Gemini + ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("📚 Processing PDF...")
    
    # Upload PDF to FastAPI server
    try:
        with open(tmp_path, "rb") as f:
            files = {"file": ("document.pdf", f, "application/pdf")}
            response = requests.post(f"{API_URL}/upload-pdf", files=files)
            response.raise_for_status()
            result = response.json()
            st.success(f"✅ {result['message']}")
    except Exception as e:
        st.error(f"Error uploading PDF: {str(e)}")
        st.stop()
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

    # Ask a question
    query = st.text_input("Ask a question about the PDF:")
    if query:
        st.info(f"🔍 Processing question: {query}")
        try:
            # Query FastAPI server
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            )
            response.raise_for_status()
            result = response.json()
            
            st.markdown("### 📖 Answer")
            st.write(result["answer"])
            
            with st.expander("View Context"):
                for i, context in enumerate(result["context"], 1):
                    st.markdown(f"**Context {i}:**")
                    st.write(context)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
