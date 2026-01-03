import os
from pathlib import Path

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from assistant import build_vector_store


if not os.path.exists("chroma_db"):
    build_vector_store()


TEXTS_DIRECTORY = "text_files"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"


def ensure_dirs():
    Path(TEXTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file):
    file_path = Path(TEXTS_DIRECTORY) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def load_documents(folder: str):
    docs = []
    for file_path in Path(folder).rglob("*"):
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())

    # normalize source metadata
    for d in docs:
        src = d.metadata.get("source")
        if not src:
            # loaders usually include source; fallback:
            d.metadata["source"] = d.metadata.get("file_path") or "unknown"
    return docs


def reindex():
    docs = load_documents(TEXTS_DIRECTORY)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # reset collection by deleting and recreating
    try:
        vs._collection.delete(where={})  # best-effort wipe
    except Exception:
        pass

    vs.add_documents(chunks)
    vs.persist()
    return len(docs), len(chunks)


def get_vector_store():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def format_sources(docs):
    seen = set()
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        label = f"{Path(src).name}"
        if page is not None:
            label += f" (page {page + 1})"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out[:6]


# ---------------- UI ----------------
ensure_dirs()
st.title("RAG Assistant — Talk to Documents")
st.caption(
    "Ask questions based only on your uploaded documents (RAG with Chroma + OpenAI).")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload TXT/PDF", type=["txt", "pdf"], accept_multiple_files=True)

    if uploaded:
        for uf in uploaded:
            save_uploaded_file(uf)
        st.success(f"Uploaded {len(uploaded)} file(s) to {TEXTS_DIRECTORY}/")

    if st.button("Reindex documents"):
        with st.spinner("Indexing..."):
            d_count, c_count = reindex()
        st.success(f"Indexed {d_count} document(s), {c_count} chunks.")

    st.divider()
    st.write("Tip: Set your API key in environment variable `OPENAI_API_KEY`.")

# session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# show history
for msg in st.session_state.chat:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            st.markdown("**Sources:**")
            for s in msg["sources"]:
                st.write(f"- {s}")

question = st.chat_input("Ask a question about your documents...")
if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vs = get_vector_store()
            retrieved = vs.similarity_search(question, k=4)

            context = "\n\n".join([d.page_content for d in retrieved])
            sources = format_sources(retrieved)

            system_rules = (
                "You answer ONLY using the provided context. "
                "If the answer is not in the context, say: \"I don't know.\" "
                "Keep it concise."
            )

            history = st.session_state.chat[-6:]
            history_text = "\n".join(
                [f"{m['role'].upper()}: {m['content']}"
                 for m in history if m["role"] in ["user", "assistant"]]
            )

            prompt = f"""
{system_rules}

Conversation:
{history_text}

Context:
{context}

Question:
{question}
"""

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        answer = llm.invoke(prompt).content

        st.markdown(answer)
        if sources:
            st.markdown("**Sources:**")
            for s in sources:
                st.write(f"- {s}")

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
