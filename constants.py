import os
from chromadb.config import Settings

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing source documents
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/source_documents"

# Define the folder for storing database
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"

MODELS_PATH = "./models"

CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)