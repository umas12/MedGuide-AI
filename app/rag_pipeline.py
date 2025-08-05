# app/rag_pipeline.py

#imports
import re
import os
import glob
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer


# Load dataset and documents
def load_documents():
    df = pd.read_csv("data/medquad.csv")
    df = df.dropna(subset=["question", "answer"])
    medquad_docs = [
        Document(
            page_content=row["answer"],
            metadata={
                "question": row["question"],
                "topic": row.get("mainEntity", ""),
                "source": row.get("URL", "MedQuAD")
            }
        )
        for ind, row in df.iterrows()
    ]
    
    pdf_docs = []
    paths = glob.glob("data/cms_pdfs/*.pdf")
    print(f"Found {len(paths)} PDF files.")
    for path in paths:
            loader = PyPDFLoader(path)#, mode="elements", strategy="fast")
            loaded = loader.load()
            pdf_docs.extend(loaded)

    return medquad_docs + pdf_docs


# Chunk documents
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


# Build or load FAISS index
# def build_index(chunks, model):
#     texts = [c.page_content for c in chunks]
#     embs = model.encode(texts, show_progress_bar=True)
#     index = faiss.IndexFlatL2(embs.shape[1])
#     index.add(np.array(embs))
    
#     faiss.write_index(index, "index/faiss_index.bin")
#     np.save("index/embeddings.npy", embs)
    
#     return index, embs

def get_or_build_index(chunks, model, index_path="index/faiss_index.bin", emb_path="index/embeddings.npy"):
    if os.path.exists(index_path) and os.path.exists(emb_path):
        print("Loading existing index and embeddings from disk...")
        index = faiss.read_index(index_path)
        embs = np.load(emb_path)
        return index, embs

    print("Building new index...")
    texts = [c.page_content for c in chunks]
    embs = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(np.array(embs))

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(emb_path, embs)

    return index, embs



# docs, chunks, embed_model, index = None, None, None, None

# def setup_pipeline():
#     global docs, chunks, embed_model, index
#     docs = load_documents()
#     chunks = chunk_documents(docs)
#     embed_model = SentenceTransformer("all-MiniLM-L6-v2")
#     index, _ = build_index(chunks, embed_model)
    
    
# Load everything
docs = load_documents()
chunks = chunk_documents(docs)
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model = SentenceTransformer("NeuML/bioclinical-modernbert-base-embeddings")
index, embs = get_or_build_index(chunks, embed_model)


# Retriever
def retrieve_top_k(question: str, k=5):
    q_emb = embed_model.encode([question])
    D, I = index.search(np.array(q_emb), k)
    
    return [chunks[i] for i in I[0]]


# Initialize QA pipeline
qa_pipe = pipeline("question-answering", 
                   model="deepset/roberta-base-squad2",
                   tokenizer="deepset/roberta-base-squad2"
                )


# Gaurdrails
def is_high_risk(q):
    return any(re.search(p, q.lower()) for p in [r"diagnos", r"treat", r"emergency", r"should I take", r"dose of", r"side effect"])


def rag_qa_guarded(question, k=5, thresh=0.2):
    if is_high_risk(question):
        return {"answer": "Please consult a qualified healthcare provider.", "score": None, "sources": []}

    docs = retrieve_top_k(question, k)
    context = "\n\n".join(d.page_content for d in docs)
    result = qa_pipe({"question": question, "context": context})

    if result["score"] < thresh:
        return {"answer": "I'm not fully confident. Please rephrase or consult a doctor.", "score": result["score"], "sources": []}

    return {
        "answer": result["answer"],
        "score": result["score"],
        "sources": [d.metadata.get("source", "unknown") for d in docs]

    }
