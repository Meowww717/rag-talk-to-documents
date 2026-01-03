# RAG Assistant — Talk to Documents

This project is a Retrieval-Augmented Generation (RAG) assistant that allows users to ask questions about their own documents and receive grounded answers with source citations.

The application demonstrates how Large Language Models (LLMs) can be combined with vector search to build reliable document-based question answering systems.

---

## 🚀 Features

- Document-based Question & Answering (RAG)
- Streamlit web interface
- Embedding-based semantic search
- Persistent vector store (Chroma)
- Source-aware answers (citations)
- Session-based conversational memory
- Support for multiple documents (TXT, PDF)

---

## 🧠 How It Works

1. Documents are loaded and split into smaller chunks.
2. Each chunk is converted into embeddings.
3. Embeddings are stored in a persistent Chroma vector database.
4. User questions are embedded and matched against relevant document chunks.
5. Retrieved context is passed to the LLM.
6. The LLM generates an answer strictly based on the retrieved context.
7. The UI displays the answer along with its source documents.

---

## 🏗 Architecture

User
→ Streamlit UI
→ Query Embedding
→ Chroma Vector Store (Semantic Search)
→ Relevant Document Chunks
→ LLM (OpenAI)
→ Answer + Citations

---

## 🧰 Tech Stack

- Python 3.10+
- Streamlit
- LangChain
- OpenAI API
- Chroma Vector Database
- PDF & Text document loaders

---

## 📂 Project Structure

TalkToDocument/
├── app.py # Streamlit application
├── assistant.py # Document ingestion and indexing
├── requirements.txt
├── text_files/ # Source documents (TXT / PDF)
├── chroma_db/ # Persistent vector database
└── README.md

---

## ⚙️ Setup & Run

1. Install dependencies
   pip install -r requirements.txt

2. Set OpenAI API key

Windows (PowerShell):
setx OPENAI_API_KEY "your_api_key_here"

3. Add documents
   Place .txt or .pdf files into the text_files/ directory.

4. Build the vector index
   python assistant.py

5. Run the application
   streamlit run app.py

Open the browser at: http://localhost:8501

🔍 Example Questions

Which country is famous for lavender fields?
What landmarks are mentioned in Ukraine?
Where can you find Mount Fuji?
Which country combines modern cities and traditional culture?

Hallucination Control
The assistant is explicitly instructed to answer only using retrieved document context.
If the answer cannot be found in the documents, it responds with:
"I don't know."
This approach ensures factual, source-grounded responses.

📌 Notes

The vector store is persistent; removing source files requires reindexing.
This project is designed as an applied, production-style RAG prototype.

License

This project is for educational and demonstration purposes.
