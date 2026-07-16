import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH")
MODEL_NAME = os.getenv("MODEL_NAME")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SENDER = os.getenv("SMTP_SENDER")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
OPIK_API_KEY = os.getenv("OPIK_API_KEY")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "dental-booking-system")
FAQ_CACHE_TTL_SECONDS = int(os.getenv("FAQ_CACHE_TTL_SECONDS", "86400"))  # 24h default
FAQ_EMBEDDING_MODEL = os.getenv("FAQ_EMBEDDING_MODEL", "text-embedding-3-small")
FAQ_SIMILARITY_THRESHOLD = float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.90"))
FAQ_INDEX_MAX_ENTRIES = int(os.getenv("FAQ_INDEX_MAX_ENTRIES", "300"))
FAQ_INDEX_KEY = os.getenv("FAQ_INDEX_KEY", "faq:index")
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_local.db")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "dental_clinic_docs")



