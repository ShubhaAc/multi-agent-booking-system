import logging
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from pinecone import Pinecone
from rag.upsert import load_chunks_with_ids


logger = logging.getLogger(__name__)

def get_pinecone_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def upsert_pinecone():
    """
    Clear ALL existing vectors currently in the Pinecone index, then
    re-ingest fresh from data/*.md. Full clear-then-rebuild — not an
    incremental upsert — so stale/orphaned chunks from a previous ingest
    never linger alongside the new ones.

    Note: index.delete(delete_all=True) targets the default namespace ("").
    If you ingest into a non-default namespace elsewhere, pass
    namespace=... to both this delete call and from_documents below.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)
    chunks, ids = load_chunks_with_ids()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True)
        logger.info("Cleared all existing vectors from Pinecone index '%s'.", PINECONE_INDEX_NAME)
    except Exception as e:
        # Pinecone raises a 404-style error if the namespace has never had
        # any vectors written to it yet (nothing to delete) — safe to
        # ignore and proceed straight to ingesting.
        logger.info("Nothing to clear in Pinecone index '%s' (%s).", PINECONE_INDEX_NAME, e)

    PineconeVectorStore.from_documents(
        chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME, ids=ids,
    )
    logger.info("Upserted %d fresh chunks into Pinecone.", len(chunks))