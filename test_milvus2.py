from dotenv import load_dotenv
load_dotenv()

from pymilvus import connections
import pymilvus.orm.connections as _orm_conn
from pymilvus.exceptions import ConnectionNotExistException

from config import MILVUS_URI, MILVUS_COLLECTION_NAME
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# One guaranteed-real handler, registered under "default".
connections.connect(alias="default", uri=MILVUS_URI)

# langchain_milvus's internal MilvusClient generates its own alias
# (e.g. "cm-...") but never registers it in the legacy ORM registry
# that Collection()/_fetch_handler read from - a known gap between
# MilvusClient and the ORM API (pymilvus-io/pymilvus#1643). Patch the
# lookup to fall back to our pre-connected "default" handler whenever
# the specific alias isn't found there.
_original_fetch_handler = _orm_conn.Connections._fetch_handler

def _patched_fetch_handler(self, alias):
    try:
        return _original_fetch_handler(self, alias)
    except ConnectionNotExistException:
        return _original_fetch_handler(self, "default")

_orm_conn.Connections._fetch_handler = _patched_fetch_handler

vs = Milvus(
    embedding_function=OpenAIEmbeddings(),
    collection_name=MILVUS_COLLECTION_NAME,
    connection_args={"uri": MILVUS_URI},
    auto_id=False,
)
print("alias used:", vs.alias)

vs.add_documents([Document(page_content="hello world test")], ids=["test-1"])
print("SUCCESS")