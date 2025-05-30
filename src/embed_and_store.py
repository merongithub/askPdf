import chromadb
from sentence_transformers import SentenceTransformer
from config import COLLECTION_NAME

# Load and prepare chunks
with open("data/chunks.txt", "r") as f:
    content = f.read().strip()
    words = content.split()
    chunk_size = 50
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

print("\nChunks to be stored:")
print("-" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)
print("-" * 50)

# Use SentenceTransformer for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# Use a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)
# Delete all items in the collection by their IDs
existing = collection.get()
if existing["ids"]:
    collection.delete(ids=existing["ids"])

# Add new chunks
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk-{i}" for i in range(len(chunks))]
)

print(f"Stored {len(chunks)} chunks in ChromaDB.")