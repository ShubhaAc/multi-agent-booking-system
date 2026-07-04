import logging
import uvicorn
from opik.integrations.langchain import OpikTracer

opik_tracer = OpikTracer()


logging.getLogger("watchfiles").setLevel(logging.WARNING)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
   
        reload_excludes=[
            "*.log", "**/*.log",
            "*.config", "**/*.config",
            "*.db", "**/*.db",
            "*.db-wal", "**/*.db-wal",
            "*.db-shm", "**/*.db-shm",
            "*.db-journal", "**/*.db-journal",
            "chroma_db/*", "**/chroma_db/*",
            "*.sqlite", "**/*.sqlite",
            "__pycache__/*", "**/__pycache__/*",
            ".opik*/*", "**/.opik*/*",
            "frontend/*", "**/frontend/**",

        ],
    )