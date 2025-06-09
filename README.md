# 🤖 RAG CV Analyzer – Match Your CV with Job Requirements using LLMs

RAG CV Analyzer is a chatbot powered by Retrieval-Augmented Generation (RAG) designed to evaluate how well your Curriculum Vitae (CV) aligns with specific Job Requirements. It leverages FastAPI as the backend to handle file uploads and user interactions, while using a local LLM through Ollama to generate private, context-aware responses—ensuring data privacy and fully local processing without reliance on external cloud services.

---

## 🚀 Features

- ✅ Upload your CV in PDF format
- ✅ Input job requirements as your query
- ✅ Receive insights on how your CV fits the job
- ✅ Runs locally for full **data privacy**
- ✅ Built with LangChain + FAISS + Ollama
- ✅ Interactive Streamlit web interface

---

## 🧠 How It Works

### RAG Pipeline

1. Load and split the CV using `RecursiveCharacterTextSplitter`
2. Create document embeddings with `OllamaEmbeddings` (`llama3`)
3. Store embeddings in a `FAISS` vector database
4. User inputs job description as a query
5. Bot retrieves relevant content and generates an answer using a local LLM (`OllamaLLM`)

---

## 🧰 Installation & Usage

### 1. Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed locally
- Download the model locally using:
  ```bash
  ollama run llama3.2
  ```
### 2. Clone the Repository

git clone https://github.com/your-username/rag-cv-analyzer.git
cd rag-cv-analyzer

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run API (Backend)
```bash
py app.py
```

### 5. Run Streamlit (Frontend)
```bash
streamlit run client.py
```