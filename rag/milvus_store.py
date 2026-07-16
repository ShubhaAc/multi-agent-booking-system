import logging
from pymilvus import connections, utility
import pymilvus.orm.connections as _orm_conn
from pymilvus.exceptions import ConnectionNotExistException
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from rag.upsert import load_chunks_with_ids
from config import MILVUS_URI, MILVUS_COLLECTION_NAME

logger = logging.getLogger(__name__)

_ALIAS = "default"
_patched = False


def _ensure_connection():

    existing_aliases = [alias for alias, _ in connections.list_connections()]
    if _ALIAS in existing_aliases:
        connections.disconnect(_ALIAS)
    connections.connect(alias=_ALIAS, uri=MILVUS_URI)

    _patch_fetch_handler_fallback()


def _patch_fetch_handler_fallback():

    global _patched
    if _patched:
        return

    original_fetch_handler = _orm_conn.Connections._fetch_handler

    def _patched_fetch_handler(self, alias):
        try:
            return original_fetch_handler(self, alias)
        except ConnectionNotExistException:
            return original_fetch_handler(self, _ALIAS)

    _orm_conn.Connections._fetch_handler = _patched_fetch_handler
    _patched = True
    logger.info("Patched pymilvus _fetch_handler for langchain-milvus compatibility.")


def get_milvus_retriever():
    embeddings = OpenAIEmbeddings()
    _ensure_connection()
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        auto_id=False,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})


def upsert_milvus():

    embeddings = OpenAIEmbeddings()
    chunks, ids = load_chunks_with_ids()

    _ensure_connection()
    if utility.has_collection(MILVUS_COLLECTION_NAME, using=_ALIAS):
        utility.drop_collection(MILVUS_COLLECTION_NAME, using=_ALIAS)
        logger.info("Dropped existing Milvus collection '%s'.", MILVUS_COLLECTION_NAME)

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        auto_id=False,
    )
    vectorstore.add_documents(chunks, ids=ids)
    logger.info(
        "Upserted %d fresh chunks into Milvus collection '%s'.",
        len(chunks), MILVUS_COLLECTION_NAME,
    )