import logging
import uvicorn
from opik.integrations.langchain import OpikTracer

opik_tracer = OpikTracer()

# Logging is configured in api.py (force=True, runs first in the actual
# worker process) — not here. See the comment at the top of api.py for why.
logging.getLogger("watchfiles").setLevel(logging.WARNING)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # Broad, defensive excludes. SQLite WAL mode writes *.db-wal / *.db-shm
        # / *.db-journal continuously — those (not the .db file itself) were
        # the actual cause of the endless "1 change detected" reload loop.
        # "**/" prefix is needed so patterns match files inside subfolders too,
        # not just the project root.
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