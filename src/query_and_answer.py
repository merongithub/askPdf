import google.generativeai as genai
import chromadb
from config import GEMINI_API_KEY, COLLECTION_NAME

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
chat_model = genai.GenerativeModel("models/gemini-1.5-pro")

# Load ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)

try:
    # Query
    query = input("Ask a question about the PDF: ")
    print(f"\nProcessing question: {query}\n")
    
    # Retrieve the most relevant chunks using embedding-based retrieval
    results = collection.query(query_texts=[query], n_results=5)
    retrieved_chunks = results['documents'][0]
    
    print("\nRetrieved chunks:")
    print("-" * 50)
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
    print("-" * 50)
    
    # Prepare context and prompt
    context = "\n".join(retrieved_chunks)
    prompt = f"""Use the following context to answer the question. If the information is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    # Get response from Gemini
    response = chat_model.generate_content(prompt)
    print("\nAnswer:\n", response.text)

except Exception as e:
    print(f"\nAn error occurred: {str(e)}")

