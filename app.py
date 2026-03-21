import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# LangChain & Vector Store
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Indian Law RAG API")

# --- Configuration ---
DATA_PATH = "Law Docs"
INDEX_PATH = "faiss_index"
# Using a slightly more capable but still small model for legal context
MODEL_ID = "openai-community/gpt2" 

# --- 1. Load & Process Documents ---
def initialize_vector_store():
    # Use CPU-optimized embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    if os.path.exists(INDEX_PATH):
        print("✅ Loading existing FAISS index...")
        # allow_dangerous_deserialization is required for loading local FAISS files safely
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("⚡ Creating new FAISS index...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return None

    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        return None

    # Chunking with overlap to preserve legal context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

# --- 2. Initialize LLM Pipeline ---
def get_llm():
    # Adding a tokenizer ensures better text handling than a raw pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.7,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=pipe)

# Global variables to hold our loaded assets
vectorstore = initialize_vector_store()
llm = get_llm()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "online", "message": "Indian Law RAG API is active 🚀"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not vectorstore:
        raise HTTPException(status_code=404, detail="No legal documents found in 'Law Docs' folder.")

    # Retrieve relevant law snippets
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(request.query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Structured prompt for legal accuracy
    prompt = f"Context from Indian Law:\n{context}\n\nQuestion: {request.query}\nAnswer accurately based on the context provided:"

    response = llm.invoke(prompt)
    
    return {
        "query": request.query,
        "answer": response.replace(prompt, "").strip(),
        "sources": [{"page": d.metadata.get("page"), "file": d.metadata.get("source")} for d in retrieved_docs]
    }

if __name__ == "__main__":
    # Port is dynamic for Render deployment
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)