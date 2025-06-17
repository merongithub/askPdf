## File: src/config.py
import os

# Configuration settings for PDF processing

# Number of words per chunk when splitting text
CHUNK_SIZE = 1000  # Adjust this value based on your needs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Chroma collection name
COLLECTION_NAME = "pdf_chunks"