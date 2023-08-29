import os
import torch

from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from constants import (PERSIST_DIRECTORY, SOURCE_DIRECTORY)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}



# Load all documents from the source_documents folder and split into chunks
def process_documents():
    documents = []
    print(f"Processing documents from {SOURCE_DIRECTORY}")
    for file in os.listdir(SOURCE_DIRECTORY):
        file_path = os.path.join(SOURCE_DIRECTORY, file)
        file_extension = os.path.splitext(file_path)[1]
        if file_extension in LOADER_MAPPING:
            loader_cls, loader_kwargs = LOADER_MAPPING[file_extension]
            loader = loader_cls(file_path, **loader_kwargs)
            loaded_documents = loader.load()
            documents.extend(loaded_documents)
        else:
            print(f"Skipping {file_path} as it has no loader")
    print(f"Loaded {len(documents)} documents")

    # Split documents into chunks
    print(f"Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")
    return texts


# Create embeddings for all documents
def main():
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl",
        model_kwargs={"device": "cuda"},)
    
    texts = process_documents()

    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None
    
    print(f"Ingestion complete. You can run queries on personalGPT now!")

if __name__ == "__main__":
    main()
