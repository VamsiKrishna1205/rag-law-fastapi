from fastapi import FastAPI
from pydantic import BaseModel
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

app = FastAPI()

# -------------------------
# 📥 Load PDFs
# -------------------------
DATA_PATH = "Law Docs"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

# -------------------------
# ✂️ Split Text
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# -------------------------
# 🧠 Embeddings
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 🗂️ FAISS DB
# -------------------------
import os

if os.path.exists("faiss_index"):
    print("✅ Loading existing FAISS index...")
    vectorstore = FAISS.load_local("faiss_index", embedding_model)
else:
    print("⚡ Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# 🤖 LLM
# -------------------------
llm = pipeline("text-generation", model="gpt2")

# -------------------------
# 📩 Request Schema
# -------------------------
class QueryRequest(BaseModel):
    query: str

# -------------------------
# 🏠 Home
# -------------------------
@app.get("/")
def home():
    return {"message": "Indian Law RAG API running 🚀"}

# -------------------------
# ❓ Ask Endpoint
# -------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.query

    retrieved_docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Answer based on Indian law context:

    {context}

    Question: {query}
    Answer:
    """

    response = llm(prompt, max_length=200)[0]["generated_text"]

    return {
        "query": query,
        "answer": response,
        "sources": [doc.metadata for doc in retrieved_docs]
    }