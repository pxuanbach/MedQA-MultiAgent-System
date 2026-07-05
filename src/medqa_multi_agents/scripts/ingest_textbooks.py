"""Ingest textbook files into the Chroma vector store.

SpaCy sentencizer splits on true sentence boundaries.
Decimal and abbreviation dots are masked before splitting to avoid false breaks.
Four lines per paragraph grouping, ten sentences per chunk.
Chunks shorter than 30 tokens are discarded.
all-mpnet-base-v2 embeddings (768-dim, CUDA-accelerated).
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from medqa_multi_agents.vectorstore.db import load_vectorstore
from medqa_multi_agents.vectorstore.embedding import load_embeddings

# ── Path: resolve from this file's location ────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent   # …/src/medqa_multi_agents/scripts/
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent  # …/Dev/MedQA-MultiAgent-System/
TEXTBOOK_DIR = _REPO_ROOT / "datasets" / "MedQA-USMLE" / "textbooks" / "en"

# ── Chunking constants ───────────────────────────────────────────────────────────
SENTENCES_PER_CHUNK = 10
MIN_TOKEN_LENGTH    = 30
BATCH_SIZE          = 5000  # Chroma max batch size


def _split_list(input_list: list, slice_size: int) -> list[list]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def _clean_for_sentencizer(text: str) -> str:
    """Mask dots that should NOT be treated as sentence boundaries."""
    # Mask decimals: 3.14 -> 3<D>14
    text = re.sub(r"(\d)\.(\d)", r"\1<D>\2", text)
    # Mask common medical abbreviations: Fig. Dr. e.g. i.e. etc.
    text = re.sub(
        r"(eFig|Fig|Table|Dr|Mr|Mrs|Ms|i\.e|e\.g|etc)\.",
        r"\1<P>",
        text,
    )
    return text


def _restore_mask(text: str) -> str:
    return text.replace("<D>", ".").replace("<P>", ".")


def _build_spacy_chunks(
    paragraphs: list[dict],
    nlp,
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
) -> list[Document]:
    """Chunk paragraphs using spaCy sentencizer."""
    chunks = []
    for item in tqdm(paragraphs, desc="Chunking textbooks"):
        cleaned = _clean_for_sentencizer(item["text"])
        doc = nlp(cleaned)
        sentences = [
            _restore_mask(str(s).strip())
            for s in doc.sents
            if str(s).strip()
        ]
        sentence_chunks = _split_list(sentences, sentences_per_chunk)
        for chunk_idx, sentence_chunk in enumerate(sentence_chunks):
            joined = " ".join(sentence_chunk).strip()
            chunk_doc = nlp(joined)
            token_count = len(chunk_doc)
            if token_count < MIN_TOKEN_LENGTH:
                continue
            chunks.append(
                Document(
                    page_content=joined,
                    metadata={
                        "source":        item["book_reference"],
                        "chunk_id":      f"{item['book_reference']}_{item['par_id']}_{chunk_idx}",
                        "num_sentences": len(sentence_chunk),
                        "token_count":   token_count,
                    },
                )
            )
    return chunks


def _load_textbooks(textbooks_dir: Path) -> list[dict]:
    """Load paragraphs from all .txt files in textbooks_dir."""
    avg_paragraph_in_page = 4
    paragraphs = []
    for txt_path in sorted(textbooks_dir.glob("*.txt")):
        lines = [
            line.strip()
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        par_id = 0
        for i in range(0, len(lines), avg_paragraph_in_page):
            group = lines[i : i + avg_paragraph_in_page]
            if group:
                paragraphs.append({
                    "book_reference": txt_path.stem,
                    "par_id":        par_id,
                    "text":          "".join(group),
                })
                par_id += 1
    return paragraphs


def ingest_textbooks(
    persist_directory: str | None = None,
    collection_name: str = "medqa_textbooks",
    textbooks_dir: Path | None = None,
    embedding_model: Embeddings | None = None,
) -> int:
    """Ingest all English textbook .txt files into Chroma.

    Four lines per paragraph, spaCy sentencizer splits on sentence boundaries,
    decimal and abbreviation dots masked before splitting, ten sentences per chunk,
    chunks under 30 tokens discarded, all-mpnet-base-v2 embeddings (768-dim, CUDA-accelerated).

    Returns the total number of chunks ingested.
    """
    from spacy.lang.en import English

    textbooks_dir  = textbooks_dir or TEXTBOOK_DIR
    embedding_model = embedding_model or load_embeddings(batch_size=32)

    # Blank English model with sentencizer — no pre-trained weights needed
    nlp = English()
    nlp.max_length = 100_000_000
    nlp.add_pipe("sentencizer")

    paragraphs = _load_textbooks(textbooks_dir)
    documents  = _build_spacy_chunks(paragraphs, nlp)

    vectorstore = load_vectorstore(
        persist_directory=persist_directory,  # None → uses DEFAULT_PERSIST_DIR from db.py
        collection_name=collection_name,
        embeddings=embedding_model,
    )

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)

    return len(documents)


if __name__ == "__main__":
    count = ingest_textbooks()
    print(f"Ingested {count} chunks into Chroma vector store.")
