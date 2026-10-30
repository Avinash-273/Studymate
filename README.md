# StudyMate – AI PDF Study Assistant

StudyMate is a Streamlit-based PDF question-answering application. Upload a PDF, process it into searchable chunks, and ask questions grounded in the document. It uses LangChain for the retrieval-augmented generation (RAG) workflow, Hugging Face sentence-transformer embeddings, FAISS for vector search, and Groq for answer generation.

## Features
- Upload and process PDF documents
- Split document text into overlapping chunks
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Retrieve relevant passages using FAISS
- Ask questions through a chat interface and receive answers based on the uploaded PDF
- Display source page references when available

## Technology stack
Python · Streamlit · LangChain · Groq · Hugging Face Embeddings · FAISS · PyPDF

## Run locally
1. Clone this repository:
   ```bash
   git clone https://github.com/Avinash-273/Studymate.git
   cd Studymate
   ```
2. Create and activate a virtual environment.
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your Groq API key.
5. Run:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
- Repository: `Avinash-273/Studymate`
- Branch: `main`
- Main file: `app.py`
- In **Advanced settings → Secrets**, add:
  ```toml
  GROQ_API_KEY = "your_rotated_groq_api_key"
  ```

Never commit `.env`, API keys, or virtual-environment folders.
