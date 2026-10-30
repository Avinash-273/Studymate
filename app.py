import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="StudyMate | PDF Study Assistant", page_icon="📚", layout="centered")
st.title("📚 StudyMate")
st.caption("Your AI study assistant. Upload a PDF and ask questions grounded in its content.")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "")
    try:
        api_key = api_key or st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets or your local .env file.")
        st.stop()
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)

def build_vectorstore(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        pdf_path = tmp.name
    try:
        pages = PyPDFLoader(pdf_path).load()
        if not pages:
            raise ValueError("No readable text was found in this PDF.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No text chunks could be created from this PDF.")
        return FAISS.from_documents(chunks, get_embeddings())
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

def format_docs(docs):
    return "\n\n".join(
        f"Source page {doc.metadata.get('page', 0) + 1}:\n{doc.page_content}"
        for doc in docs
    )

with st.sidebar:
    st.header("Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_pdf:
        st.success(f"Selected: {uploaded_pdf.name}")
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading, splitting, embedding, and indexing your PDF..."):
                try:
                    st.session_state.vectorstore = build_vectorstore(uploaded_pdf)
                    st.session_state.pdf_name = uploaded_pdf.name
                    st.session_state.messages = []
                    st.success("PDF processed. You can now ask questions.")
                except Exception as exc:
                    st.error(f"Could not process the PDF: {exc}")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Read the uploaded PDF\n2. Split text into chunks\n3. Create embeddings\n4. Retrieve relevant passages with FAISS\n5. Generate a grounded answer with Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.vectorstore is None:
    st.info("Upload a PDF in the sidebar and click **Process PDF** to begin.")
    st.stop()

st.caption(f"Currently studying: {st.session_state.get('pdf_name', 'Uploaded document')}")
question = st.chat_input("Ask a question from your PDF...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDF and preparing an answer..."):
            try:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are StudyMate, a helpful study assistant.
Answer ONLY using the provided PDF context.
If the answer is absent, say: "I could not find this information in the uploaded PDF."
Explain clearly and simply. Mention relevant source page numbers when available. Do not invent information.
Context:\n{context}"""),
                    ("human", "{question}")
                ])
                chain = (
                    RunnableParallel(
                        context=retriever | format_docs,
                        question=RunnablePassthrough()
                    )
                    | prompt
                    | get_llm()
                    | StrOutputParser()
                )
                answer = chain.invoke(question)
            except Exception as exc:
                answer = f"Sorry, something went wrong: {exc}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
