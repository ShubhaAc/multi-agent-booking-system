import logging
import os
import sys
import uvicorn
from opik.integrations.langchain import OpikTracer

# 1. Centralized System-Wide Logging Configuration
def setup_global_logging():
    # Create the log directory if it doesn't exist
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "app.log")

    # Define a detailed format for file logging (includes timestamps, modules, line numbers)
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Define a cleaner, scannable format for terminal output
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
        datefmt='%H:%M:%S'
    )

    # File Handler (Appends everything down to DEBUG level)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console/Terminal Handler (Streams clean logs down to INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure the Root Logger (The parent of all loggers across your entire application)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything globally; handlers will filter it
    
    # Clear any existing handlers to prevent duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 2. Silence Noisy Third-Party Frameworks
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


# Initialize logging immediately before any sub-modules create their child loggers
setup_global_logging()

# Initialize Opik tracing after logging is wired up
opik_tracer = OpikTracer()


if __name__ == "__main__":
    logging.info("Starting Multi-Agent Booking System...")
    
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