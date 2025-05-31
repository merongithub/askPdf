# PDF Question Answering Application

This application allows you to ask questions about a PDF document and receive answers based on the content of the PDF. It uses ChromaDB for storing and retrieving document chunks and the Gemini model for generating answers.

## Features

- **PDF Processing**: The application reads a PDF file and splits it into manageable chunks.
- **Embedding Generation**: It uses the SentenceTransformer model to generate embeddings for each chunk.
- **ChromaDB Storage**: The chunks and their embeddings are stored in ChromaDB for efficient retrieval.
- **Question Answering**: Users can ask questions about the PDF, and the application retrieves relevant chunks and uses the Gemini model to generate answers.

## Prerequisites

- Python 3.6 or higher
- Required Python packages (install using `pip`):
  - `chromadb`
  - `google-generativeai`
  - `PyMuPDF`
  - `sentence-transformers`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/merongithub/askPdf.git
   cd askPdf
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   - Create a `.env` file in the project root and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

### Step 1: Process the PDF

Run the `embed_and_store.py` script to process the PDF and store the chunks in ChromaDB:

```bash
python3 src/embed_and_store.py
```

This script will:
- Read the PDF file from `data/sample.pdf`.
- Split the content into chunks of 50 words each.
- Generate embeddings for each chunk.
- Store the chunks and embeddings in ChromaDB.

### Step 2: Ask Questions

Run the `query_and_answer.py` script to ask questions about the PDF:

```bash
python3 src/query_and_answer.py
```

The script will:
- Prompt you to enter a question.
- Retrieve the most relevant chunks from ChromaDB.
- Use the Gemini model to generate an answer based on the retrieved chunks.

## Project Structure

- `src/embed_and_store.py`: Processes the PDF and stores chunks in ChromaDB.
- `src/query_and_answer.py`: Retrieves chunks and generates answers to questions.
- `data/sample.pdf`: The PDF file to be processed.
- `data/chunks.txt`: The processed chunks from the PDF.

## Troubleshooting

- **Empty Retrieval**: If the retrieval returns empty chunks, ensure that the PDF is processed correctly and that the ChromaDB collection is populated.
- **API Key Issues**: Verify that your Gemini API key is correctly set in the `.env` file.

## Launching the Streamlit App

To launch the Streamlit app, follow these steps:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

3. Open your web browser and navigate to the URL provided in the terminal (usually http://localhost:8501).

4. Upload a PDF file and ask questions about its content using the interface.

## Additional Information

- The app uses Streamlit for the user interface, allowing you to interact with the PDF content using a web-based interface.
- Ensure your Google API key is correctly set in `config.py` to use the Gemini model for generating answers.

