"""Persistent ChromaDB client."""

from pathlib import Path

from langchain_chroma import Chroma

from medqa_multi_agents.vectorstore.embedding import load_embeddings

DEFAULT_PERSIST_DIR = str(Path(__file__).parent / ".chroma_db")
DEFAULT_COLLECTION_NAME = "medqa_textbooks"


def load_vectorstore(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embeddings=None,
) -> Chroma:
    """Load or create a persistent Chroma vector store.

    If the persist_directory does not exist, Chroma will create it.
    """
    if embeddings is None:
        embeddings = load_embeddings()

    return Chroma(
        client=None,  # Let LangChain create a persistent client
        persist_directory=persist_directory or DEFAULT_PERSIST_DIR,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
