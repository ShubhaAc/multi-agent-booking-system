import hashlib
import logging

from rag.ingest import load_documents

logger = logging.getLogger(__name__)

def _stable_id(doc, index:int) -> str:
 

    source = doc.metadata.get("source", "unknown")
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{index}:{digest}"

def load_chunks_with_ids():
    chunks = load_documents()
    ids = [_stable_id(doc, i) for i, doc in enumerate(chunks)]
    return chunks, ids