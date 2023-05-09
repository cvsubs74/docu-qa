import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone


class VectorOperations:
    def __init__(self, api_key, environment):
        pinecone.init(
            api_key=api_key,
            environment=environment
        )

    @staticmethod
    def vectorize(index_name, namespace, document):
        # Index
        index = pinecone.Index(index_name)
        # Embeddings
        embeddings = OpenAIEmbeddings(chunk_size=1)
        # Chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(document.get_content())
        # vectorize
        vector_store = Pinecone.from_texts(
            [t.page_content for t in chunks], embeddings,
            index_name=index_name, namespace=namespace)
        return vector_store

    @staticmethod
    def remove_doc(index_name, filename):
        # Index
        index = pinecone.Index(index_name)
        index.delete(delete_all=True, namespace=filename)
