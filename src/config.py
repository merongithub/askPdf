## File: src/config.py
import os

# Configuration settings for PDF processing

# Number of words per chunk when splitting text
CHUNK_SIZE = 1000  # Adjust this value based on your needs

# Gemini API Key (replace with your actual key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyACdCxTY--EAcKbDwkMbo4KpsiD3YypJV8")

# Chroma collection name
COLLECTION_NAME = "pdf_chunks"