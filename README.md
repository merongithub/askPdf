# AskPDF

<p align="center">
  <img src="assets/askPdf.png" alt="AskPDF Application Screenshot" width="300">
</p>

# PDF Question Answering Application

A Streamlit application that allows users to upload PDF documents and ask questions about their content using Google's Gemini AI and Vertex AI Vector Search.

## Features

- 📄 PDF document upload and processing
- 🔍 Semantic search using Vertex AI Vector Search
- 💬 Question answering using Google's Gemini AI
- 🎯 Accurate answers based on document content
- 🔒 Secure handling of API keys and credentials

## Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account
- Gemini API key
- Vertex AI enabled in your GCP project

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/askPdf.git
cd askPdf
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your credentials:
   - Create a `.streamlit/secrets.toml` file with the following structure:
   ```toml
   # Google Cloud Project Configuration
   GCP_PROJECT_ID = "your-project-id"
   GCP_LOCATION = "us-central1"
   VERTEX_AI_INDEX_ID = "your-index-id"
   VERTEX_AI_ENDPOINT_ID = "your-endpoint-id"
   GEMINI_API_KEY = "your-gemini-api-key"

   # Service Account Credentials
   [gcp_service_account]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = """your-private-key"""
   client_email = "your-service-account-email"
   client_id = "your-client-id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "your-client-cert-url"
   universe_domain = "googleapis.com"

   # Vertex AI Index Configuration
   [vertex_ai_index]
   dimensions = 384
   algorithm_config = { "bruteForceConfig" = {} }
   distance_measure_type = "DOT_PRODUCT_DISTANCE"

   # Vertex AI Streaming Index Configuration
   [vertex_ai_stream_index]
   dimensions = 384
   algorithm_config = { "bruteForceConfig" = {} }
   distance_measure_type = "DOT_PRODUCT_DISTANCE"
   shard_size = "SHARD_SIZE_MEDIUM"
   ```

## Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a PDF document

4. Ask questions about the document content

## Project Structure

```
askPdf/
├── src/
│   └── app.py              # Main application code
├── .streamlit/
│   └── secrets.toml        # API keys and configuration
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Configuration

The application uses Streamlit secrets for configuration. All sensitive information and configuration settings are stored in `.streamlit/secrets.toml`, including:

- Google Cloud Project settings
- Service Account credentials
- Vertex AI index configurations
- API keys

## Dependencies

- streamlit: Web application framework
- PyMuPDF (fitz): PDF processing
- sentence-transformers: Text embedding generation
- google-generativeai: Gemini API integration
- google-cloud-aiplatform: Vertex AI integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


- Run tests:
  ```bash
  python -m unittest tests/test_pdf_processing.py
  ```
- Google Gemini AI for the language model
- Vertex AI for vector search capabilities
- Streamlit for the web interface framework

