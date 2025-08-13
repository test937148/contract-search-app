# app.py

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import re

# Optional: suppress warnings about broken distributions
import warnings
warnings.filterwarnings("ignore", message="Ignoring invalid distribution")

# --- Helper functions ---

def chunk_text(text, max_chunk_size=500):
    """Split text into smaller chunks for better semantic search."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query, docs, top_n=3):
    query_emb = embed_model.encode(query)
    sims = []
    for doc in docs:
        sim = cosine_similarity(query_emb, doc['embedding'])
        sims.append((doc, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

def extract_paragraph(text, answer):
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        if answer.lower() in para.lower():
            return para.strip()
    return text[:500] + "..."

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# --- Streamlit interface ---
st.title("📄 AI Contract Search & Q&A")

uploaded_files = st.file_uploader(
    "Upload contract PDFs", 
    accept_multiple_files=True, 
    type=['pdf']
)

if uploaded_files:
    st.info("Processing uploaded contracts...")

    # Step 1: Extract & chunk
    documents = []
    chunk_id = 0
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        chunks = chunk_text(text)
        for chunk in chunks:
            documents.append({
                "doc_id": i,
                "filename": uploaded_file.name,
                "chunk_id": chunk_id,
                "text": chunk
            })
            chunk_id += 1

    st.success(f"{len(documents)} text chunks created from {len(uploaded_files)} documents.")

    # Step 2: Load models (cache to speed up reloads)
    @st.cache_resource
    def load_models():
        em = SentenceTransformer('all-MiniLM-L6-v2')
        qa = pipeline('question-answering', model='deepset/roberta-base-squad2')
        return em, qa

    embed_model, qa_pipeline = load_models()

    # Step 3: Embed all chunks (cache)
    @st.cache_data(show_spinner=False)
    def embed_texts(docs):
        for doc in docs:
            doc['embedding'] = embed_model.encode(doc['text'])
        return docs

    documents = embed_texts(documents)

    # Step 4: User query
    query = st.text_input("Ask a question about your contracts:")

    if query:
        results = semantic_search(query, documents, top_n=3)
        st.markdown(f"### Top {len(results)} Results:")
        for rank, (doc, score) in enumerate(results, start=1):
            answer = answer_question(query, doc['text'])
            context = extract_paragraph(doc['text'], answer)
            st.write(f"**Result #{rank} | Score: {score:.3f} | Document: {doc['filename']}**")
            st.write(f"> {context}")
            st.markdown(f"**Answer:** {answer}")
            st.markdown("---")
