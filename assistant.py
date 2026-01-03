import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

TEXTS_DIRECTORY = "text_files"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"


def load_documents(folder: str):
    docs = []
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for file_path in p.rglob("*"):
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())

    # add source metadata (useful for citations)
    for d in docs:
        if "source" not in d.metadata:
            d.metadata["source"] = d.metadata.get(
                "file_path") or d.metadata.get("source", "unknown")

    return docs


def build_vector_store():
    docs = load_documents(TEXTS_DIRECTORY)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vector_store.add_documents(chunks)

    print(f"✅ Loaded {len(docs)} docs, created {len(chunks)} chunks")
    print(f"✅ Persisted Chroma DB to: {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_store()
