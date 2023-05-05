import os
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


class Document:
    def __init__(self, uploaded_file):
        temp_dir = tempfile.TemporaryDirectory()
        filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(filepath)
        self.contents = loader.load()

    def split_into_chunks(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(self.contents)
