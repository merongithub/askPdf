# PDF Question Answering Application

This application allows you to ask questions about a PDF document and receive answers based on the content of the PDF. It uses ChromaDB for storing and retrieving document chunks and the Gemini model for generating answers.

## Features

- **PDF Processing**: The application reads a PDF file and splits it into manageable chunks.
- **Embedding Generation**: It uses the SentenceTransformer model to generate embeddings for each chunk.
- **ChromaDB Storage**: The chunks and their embeddings are stored in ChromaDB for efficient retrieval.
- **Question Answering**: Users can ask questions about the PDF, and the application retrieves relevant chunks and uses the Gemini model to generate answers.
- **Web Interface**: A user-friendly Streamlit interface for uploading PDFs and asking questions.

## Prerequisites

- Python 3.6 or higher
- Required Python packages (install using `pip`):
  ```bash
  pip install -r requirements.txt
  ```

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
   - Create a `.streamlit/secrets.toml` file and add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```

## Usage

### Using the Streamlit App (Recommended)

1. Launch the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to the URL provided in the terminal (usually http://localhost:8501).

3. Upload a PDF file and ask questions about its content using the interface.

### Using Command Line Tools

#### Process PDF and Store Chunks

Run the `embed_and_store.py` script to process the PDF and store the chunks in ChromaDB:

```bash
python3 src/embed_and_store.py
```

This script will:
- Read the PDF file from `data/sample.pdf`
- Split the content into chunks of 50 words each
- Generate embeddings for each chunk
- Store the chunks and embeddings in ChromaDB

#### Query PDF Content

Run the `query_and_answer.py` script to ask questions about the PDF:

```bash
python3 src/query_and_answer.py
```

The script will:
- Prompt you to enter a question
- Retrieve the most relevant chunks from ChromaDB
- Use the Gemini model to generate an answer based on the retrieved chunks

## Project Structure

```
askPdf/
├── .streamlit/
│   └── secrets.toml        # API key configuration
├── data/
│   ├── sample.pdf         # Sample PDF file
│   └── chunks.txt         # Processed chunks
├── src/
│   ├── app.py            # Streamlit web application
│   ├── embed_and_store.py # PDF processing and storage
│   └── query_and_answer.py # Question answering
├── tests/
│   └── test_pdf_processing.py # Unit tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Troubleshooting

- **Empty Retrieval**: If the retrieval returns empty chunks, ensure that the PDF is processed correctly and that the ChromaDB collection is populated.
- **API Key Issues**: Verify that your Gemini API key is correctly set in the `.streamlit/secrets.toml` file.
- **Memory Issues**: The application uses in-memory ChromaDB by default. For large PDFs, ensure you have sufficient RAM available.

## Development

- Run tests:
  ```bash
  python -m unittest tests/test_pdf_processing.py
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## How to set up 

pyenv install 3.10.12
pyenv local 3.10.12
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
If using Streamlit Cloud, just set python_version = 3.10 in runtime.txt: