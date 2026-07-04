import hashlib
import logging

from rag.ingest import load_documents

logger = logging.getLogger(__name__)

def _stable_id(doc, index:int) -> str:
    """
    Deterministic ID from source path + chunk index + a content hash.
    Kept even with clear-then-rebuild, because Pinecone/Milvus still need
    an id per vector on write — this just guarantees the same source chunk
    gets the same id across rebuilds, which is handy for debugging/logs
    even though the clear step means duplicates were never the risk here.
    
    """

    source = doc.metadata.get("source", "unknown")
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{index}:{digest}"

def load_chunks_with_ids():
    chunks = load_documents()
    ids = [_stable_id(doc, i) for i, doc in enumerate(chunks)]
    return chunks, ids