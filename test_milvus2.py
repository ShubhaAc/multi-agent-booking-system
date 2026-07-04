from dotenv import load_dotenv
load_dotenv()

from pymilvus import connections
import pymilvus.orm.connections as _orm_conn
from pymilvus.exceptions import ConnectionNotExistException

from config import MILVUS_URI, MILVUS_COLLECTION_NAME
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

connections.connect(alias="default", uri=MILVUS_URI)

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