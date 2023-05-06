import os

import pinecone
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores import Pinecone

from document import Document
from docuqa import DocuQA

index_name = "docuqa"


def main():
    # Set page config
    docu_qa = DocuQA()
    docu_qa.set_page_config()
    # Load llm
    llm = load_llm()
    # Initialize the vector store
    initialize_vector_store()
    # Display introduction
    docu_qa.display_introduction()
    # Get uploaded document
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf"])
    if uploaded_file:
        with st.spinner("Please wait..."):
            vector_store = vectorize_and_save(docu_qa, uploaded_file)
        # query
        docu_qa.user_query(vector_store, llm)
    # Display usage statistics
    docu_qa.display_usage_stats()


@st.cache_resource
def initialize_vector_store():
    # initialize pinecone
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"]
    )


@st.cache_resource
def load_llm():
    os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    return AzureOpenAI(temperature=0.9, deployment_name="text-davinci-003-dev1", model_name="text-davinci-003")


@st.cache_resource(show_spinner=False)
def vectorize_and_save(_docu_qa, uploaded_file):
    # Index
    index = pinecone.Index(index_name)
    # split the documents into chunks
    texts = Document(uploaded_file).split_into_chunks()
    # Remove the document first
    remove_doc(index, uploaded_file)
    # Add document counter
    _docu_qa.add_document_count_and_size(size=uploaded_file.size / (1024 * 1024))
    return vectorize(texts, uploaded_file)


def vectorize(texts, uploaded_file):
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(chunk_size=1)
    # Re-vectorize it
    vector_store = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings,
        index_name=index_name, namespace=uploaded_file.name)
    return vector_store


def remove_doc(index, uploaded_file):
    index.delete(delete_all=True, namespace=uploaded_file.name)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
