import json
import logging
import time
from upstash_redis.asyncio import Redis
from config import UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, FAQ_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

_redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)

FAQ_INDEX_KEY = "faq:index"

def _key(sender_id: str) -> str:
    return f"conv:{sender_id}"


async def get_conversation(sender_id: str) -> dict:
    raw = await _redis.get(_key(sender_id))
    if not raw:
        return {"history": [], "state": {}}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Corrupt conversation JSON for sender %s, resetting.", sender_id)
        return {"history": [], "state": {}}


async def save_conversation(sender_id: str, history: list[dict], state: dict) -> None:
    await _redis.set(_key(sender_id), json.dumps({"history": history, "state": state}))




async def get_faq_index() -> list[dict]:
    raw = await _redis.get(FAQ_INDEX_KEY)
    if not raw:
        return []
    try:
        index = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Corrupt FAQ index JSON, resetting.")
        return []

    cutoff = time.time() - FAQ_CACHE_TTL_SECONDS
    fresh = [e for e in index if e.get("created_at", 0) >= cutoff]
    if len(fresh) != len(index):
        logger.info("Dropped %d stale FAQ entries (older than %ds)", len(index) - len(fresh), FAQ_CACHE_TTL_SECONDS)
        await _redis.set(FAQ_INDEX_KEY, json.dumps(fresh))
    return fresh


async def add_faq_entry(question: str, answer: str, embedding: list[float], max_entries: int) -> None:
    index = await get_faq_index()
    index.append({"question": question, "answer": answer, "embedding": embedding, "created_at": time.time()})
    if len(index) > max_entries:
        index = index[-max_entries:]  # drop oldest first
    await _redis.set(FAQ_INDEX_KEY, json.dumps(index))
    logger.info("FAQ index STORE for %r (index size now %d)", question, len(index))