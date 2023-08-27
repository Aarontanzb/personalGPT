import os

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
    # Add more mappings for other file extensions and loaders as needed
}


def process_documents():
    source_folder = "source_documents"
    documents = []
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        file_extension = os.path.splitext(file_path)[1]
        if file_extension in LOADER_MAPPING:
            loader_cls, loader_kwargs = LOADER_MAPPING[file_extension]
            loader = loader_cls(file_path, **loader_kwargs)
            loaded_documents = loader.load()
            documents.extend(loaded_documents)
    return documents

