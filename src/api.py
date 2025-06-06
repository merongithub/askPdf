from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import chromadb
from tempfile import NamedTemporaryFile
import os
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="PDF Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Config
# Number of words per chunk when splitting PDF text
# A smaller chunk size (50 words) means:
# - More granular text segments for searching
# - Potentially more precise context matching
# - But also more chunks to process and store
CHUNK_SIZE = 50
COLLECTION_NAME = "pdf_chunks"

# Initialize models
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    chat_model = genai.GenerativeModel("models/gemini-1.5-pro")
except Exception as e:
    raise Exception(f"Failed to initialize models: {str(e)}")

# Initialize ChromaDB
try:
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    raise Exception(f"Failed to initialize ChromaDB: {str(e)}")

# Utility: Read and chunk PDF
def read_and_chunk_pdf(pdf_file):
    try:
        # Try to open the PDF file
        doc = fitz.open(pdf_file)
        
        # Check if the document is empty
        if doc.page_count == 0:
            raise ValueError("The PDF file is empty")
            
        # Check if the document is encrypted
        if doc.is_encrypted:
            raise ValueError("The PDF file is password-protected")
            
        # Extract text from all pages
        text = ""
        for page in doc:
            page_text = page.get_text()
            if not page_text.strip():
                continue  # Skip empty pages
            text += page_text + "\n"
            
        # Check if we got any text
        if not text.strip():
            raise ValueError("No readable text found in the PDF")
            
        # Split into chunks
        words = text.split()
        if not words:
            raise ValueError("No words found in the PDF")
            
        chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
        
        # Close the document
        doc.close()
        
        return chunks
    except fitz.fitz.EmptyFileError:
        raise ValueError("The PDF file is empty or corrupted")
    except fitz.fitz.FileDataError:
        raise ValueError("The file is not a valid PDF")
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

# Request model for querying the PDF
class QueryRequest(BaseModel):
    # The user's question/query about the PDF content
    query: str

# Response model containing the answer and context
class QueryResponse(BaseModel):
    # The generated answer from the Gemini model
    answer: str
    # List of relevant text chunks from the PDF that were used as context
    context: List[str]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
            
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process PDF
            chunks = read_and_chunk_pdf(tmp_path)
            if not chunks:
                raise HTTPException(status_code=400, detail="No content could be extracted from the PDF")
                
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

            return {"message": f"Successfully processed PDF with {len(chunks)} chunks"}
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    try:
        results = collection.query(
            query_texts=[request.query],
            n_results=5
        )
        context = results['documents'][0]
        prompt = f"""Use the following context to answer the question. If the information is not in the context, say so.\n\nContext:{context}\n\nQuestion: {request.query}\n\nAnswer:"""
        response = chat_model.generate_content(prompt)
        
        return QueryResponse(
            answer=response.text,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 