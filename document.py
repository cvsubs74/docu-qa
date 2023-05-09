import os
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


class Document:
    def __init__(self, uploaded_file):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        loader = PyPDFLoader(temp_file.name)
        self.contents = loader.load()
        os.unlink(temp_file.name)

    def get_content(self):
        return self.contents
