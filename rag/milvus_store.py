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
    """
    
    Guarantee a live, freshly-connected handler under _ALIAS in the legacy
    ORM registry, and make sure the pymilvus/langchain-milvus compatibility
    patch below is installed exactly once.

    Always disconnects and reconnects rather than trusting
    list_connections() - a connection registered from a different thread
    (e.g. established once at FastAPI startup on the main thread, then
    reused later inside run_in_threadpool's worker thread) can be listed
    as "existing" without actually being usable from the calling thread.
    This function runs on a rare, admin-triggered path, so the extra
    reconnect cost here is negligible - trading a little redundancy for
    not silently hitting that gap again.
    """
    existing_aliases = [alias for alias, _ in connections.list_connections()]
    if _ALIAS in existing_aliases:
        connections.disconnect(_ALIAS)
    connections.connect(alias=_ALIAS, uri=MILVUS_URI)

    _patch_fetch_handler_fallback()


def _patch_fetch_handler_fallback():
    """
    Compatibility shim for pymilvus>=2.6 + langchain-milvus 0.3.3.

    langchain_milvus builds its own internal MilvusClient and uses its
    auto-generated alias (e.g. "cm-xxxxx") for self.alias, but never
    registers that alias in the legacy ORM `connections` registry.
    langchain_milvus's `self.col` property still calls the deprecated
    `Collection(name, using=self.alias)`, which looks the alias up ONLY
    in that ORM registry via `connections._fetch_handler(alias)` - so it
    always raises ConnectionNotExistException, even though a working
    connection genuinely exists (just under MilvusClient's own separate
    connection manager). This is a known upstream gap between the two
    APIs (see pymilvus-io/pymilvus#1643), not anything wrong with our
    connection setup.

    Fix: patch `_fetch_handler` so that if the specific alias isn't found
    in the ORM registry, it falls back to our pre-connected `_ALIAS`
    handler instead of raising. Idempotent - safe to call repeatedly.
    """
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
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def upsert_milvus():
    """
    Clear the existing Milvus collection entirely, then rebuild it fresh
    from data/*.md.
    """
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