# RAG PDF QA System

A **Retrieval-Augmented Generation (RAG)** system for querying PDF documents with context-aware answers from a **Large Language Model (LLM)**.  
Supports uploading PDFs, extracting text, indexing with embeddings, and answering questions using semantic search.

Built with **Django**, **FAISS**, **LangChain**, and **Hugging Face Transformers**.

---

## 🚀 Features

- Upload PDF documents and extract text automatically  
- Chunk text and generate embeddings for semantic search  
- Persist embeddings with FAISS for fast retrieval  
- Ask questions about uploaded PDFs with context-aware answers  
- Dockerized for easy setup and deployment  

---

## 🛠 Tech Stack

- **Backend:** Django, Django REST Framework  
- **Vector Store:** FAISS  
- **Embeddings:** HuggingFaceEmbeddings (`sentence-transformers/all-MiniLM-L6-v2`)  
- **LLM:** Hugging Face Transformers pipeline (`google/flan-t5-base`)  
- **Python Libraries:** `python-dotenv`, `django`, `djangorestframework`, `langchain`, `langchain-huggingface`, `pyPDF2`, `sentence-transformers`, `faiss-cpu`, `accelerate`, `psycopg2`, `pdf2image`, `pytesseract`  
- **Database:** PostgreSQL  
- **Containerization:** Docker, Docker Compose  

---

## ⚙️ Setup & Installation

1. **Clone the repository**  
```bash
git clone https://github.com/KnuppLucas/tractian-challenge-ai-engineer
cd tractian-challenge-ai-engineer
```

2. **Create `.env` file** with the following variables:
```env
HUGGINGFACEHUB_API_TOKEN=your_token
LLM_MODEL=google/flan-t5-base
POSTGRES_USER=tractian_user
POSTGRES_PASSWORD=tractian_pass
POSTGRES_DB=tractian_db
POSTGRES_HOST=db
```

3. **Build and run Docker containers**  
```bash
docker-compose up --build
```

4. **Apply Django migrations**  
```bash
docker-compose exec web python manage.py migrate
```

---

## 📝 API Endpoints

### Upload PDFs
**POST** `/documents/`  
- **Content-Type:** multipart/form-data  
- **Body:** `files` (one or more PDFs)

**Example cURL:**
```curl
curl --location 'http://127.0.0.1:8000/api/documents/' \
--form 'files=@"/path/to/file1.pdf"' \
--form 'files=@"/path/to/file2.pdf"'
```

**Response:**
```json
{
  "message": "Documents processed successfully",
  "documents_indexed": 4,
  "total_chunks": 614
}
```

---

### Ask a question
**POST** `/question/`  
- **Content-Type:** application/json  
- **Body:**
```json
{
  "question": "What is the power consumption of the motor?"
}
```

**Example cURL:**
```curl
curl --location 'http://127.0.0.1:8000/api/question/' \
--header 'Content-Type: application/json' \
--data '{
   "question": "What is the power consumption of the motor?"
}'
```

**Response:**
```json
{
  "answer": "d Pmec = (W) t 490 P1 = = 245 W ...",
  "references": [
    "voltage applied to the motor and yield ...",
    "aplicada e se calcula dividindo a energia ...",
    "Figura 3.16 - Exemplo das características ..."
  ]
}
```

---

## 🏗 Architecture Overview

1. PDF Upload → Text extraction  
2. Chunking & Embedding → Store vectors in FAISS / Postgres  
3. Retrieval → Get relevant chunks  
4. LLM → Generate answer from retrieved context  

**Services involved:**
- DocumentProcessorService: Extracts text and splits into chunks  
- EmbeddingService: Generates embeddings and persists in FAISS/Postgres  
- RAGService: Retrieves relevant chunks and generates LLM answers  

---

## 📬 Notes

- Hugging Face API token required for embeddings and LLM  
- FAISS index is persisted for faster reloads (`faiss_index/`)  
- Lightweight LLM recommended for local/dev environments  
- Larger LLMs require GPU and proper acceleration  

---

## 🔗 References

- LangChain Docs: https://www.langchain.com/docs/  
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index  
- FAISS: https://faiss.ai/
