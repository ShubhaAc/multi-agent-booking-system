import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rag.ingest import load_documents
from rag.upsert import load_chunks_with_ids
from config import CHROMA_PERSIST_DIR
import os


logger = logging.getLogger(__name__)

def get_chroma_retriever():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        logger.info("Loading existing Chroma  index.")
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        logger.info("Building new Chroma index.")
        chunks = load_documents()
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
    return vectorstore.as_retriever(search_kwargs={"k": 10})


def upsert_chroma():
    """
    Clear ALL existing chunks currently in Chroma, then re-ingest fresh from
    data/.md. Full clear-then-rebuild — not an incremental upsert — so
    stale/orphaned chunks from a previous ingest (e.g. a source doc that
    was deleted, renamed, or re-chunked differently) never linger in the
    index alongside the new ones.
    """
    embeddings = OpenAIEmbeddings()
    chunks, ids = load_chunks_with_ids()
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

    existing_ids = (vectorstore.get() or {}).get("ids") or []
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
        logger.info("Cleared %d old chunks from Chroma.", len(existing_ids))

    vectorstore.add_documents(chunks, ids=ids)
    logger.info("Upserted %d fresh chunks into Chroma.", len(chunks))