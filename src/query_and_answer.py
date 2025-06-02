import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import GEMINI_API_KEY, COLLECTION_NAME

# Initialize models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key=GEMINI_API_KEY)
chat_model = genai.GenerativeModel("models/gemini-1.5-pro")

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(COLLECTION_NAME)

def query_pdf(question: str) -> str:
    # Get relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    # Prepare context and prompt
    context = "\n".join(results['documents'][0])
    prompt = f"""Use the following context to answer the question. If the information is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer
    response = chat_model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    question = input("Enter your question about the PDF: ")
    answer = query_pdf(question)
    print("\nAnswer:", answer) 