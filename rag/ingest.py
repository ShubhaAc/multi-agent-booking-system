import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_documents(data_dir: str = "data"):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks