"""SentenceTransformer embeddings wrapper for LangChain Chroma."""

import torch
from langchain_huggingface import HuggingFaceEmbeddings

# all-mpnet-base-v2 (768-dim) — higher quality embeddings, fast on GPU
DEFAULT_MODEL = "all-mpnet-base-v2"


def load_embeddings(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance, using CUDA if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size},
    )
