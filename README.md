# VFSTR R22 Academic Regulations RAG Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions from the VFSTR B.Tech R22 Academic Regulations document.

## Project Overview

The application retrieves relevant sections from the VFSTR R22 regulations PDF and uses Google Gemini to generate a concise answer based only on the retrieved context.

## Architecture

PDF Document
→ Text Extraction (PyPDF)
→ Text Chunking
→ Sentence Transformer Embeddings
→ FAISS Vector Search
→ Relevant Context Retrieval
→ Google Gemini
→ Answer + Source Pages

## Technologies

- Python
- Streamlit
- PyPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Google Gemini
- LangChain text splitters

## Project Structure

```text
University-RAG-Assistant/
├── documents/
│   └── vfstr_r22_regulations.pdf
├── app.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_api_key_here
```

4. Start the application:

```bash
streamlit run app.py
```

5. Open the local Streamlit URL shown in the terminal, normally `http://localhost:8501`.

## Example Questions

- What are the attendance requirements for students?
- How many credits are required for the B.Tech degree?
- What are the rules for condonation of attendance shortage?

## RAG Behavior

The assistant is instructed to answer only from the retrieved VFSTR R22 regulations context. If the requested information is not present in the document, it responds that the information could not be found in the VFSTR R22 regulations.

## Author

K. Shakir

B.Tech Artificial Intelligence and Machine Learning
VFSTR
