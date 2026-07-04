import logging
from dotenv import load_dotenv
load_dotenv()

from rag.ingest import load_documents
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_INDEX_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ingest():
    logger.info("Loading documents...")
    chunks = load_documents()
    logger.info("Loaded %d chunks.", len(chunks))
    logger.info("Uploading to Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)
    PineconeVectorStore.from_documents(chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)
    logger.info("Done.")

if __name__ == "__main__":
    ingest()