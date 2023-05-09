import os

import pinecone
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores import Pinecone

from document import Document
from docuqa import DocuQA
from llm import LLM
from vectoroperations import VectorOperations

index_name = "docuqa"


def main():
    # Set page config
    DocuQA.set_page_config()
    # Load llm
    llm = load_llm()
    # Initialize the vector store
    vector_ops = vector_operations()
    # Display introduction
    DocuQA.display_introduction()
    # Get uploaded document
    docu_qa = DocuQA()
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf"])
    # File uploaded?
    if uploaded_file:
        document = Document(uploaded_file)
        vector_store = None
        with st.spinner("Please wait..."):
            vector_store = create_vectors(docu_qa, document, uploaded_file, vector_ops)
        # query
        docu_qa.user_query(vector_store, llm)
    # Display usage statistics
    docu_qa.display_usage_stats()


@st.cache_resource(show_spinner=False)
def create_vectors(_docu_qa, _document, uploaded_file, _vector_ops):
    _docu_qa.add_document_count_and_size(uploaded_file.size / (1024 * 1024))
    return _vector_ops.vectorize(index_name, uploaded_file.name, _document)


@st.cache_resource
def vector_operations():
    return VectorOperations(st.secrets["PINECONE_API_KEY"], st.secrets["PINECONE_ENV"])


@st.cache_resource
def load_llm():
    return LLM(st.secrets["OPENAI_API_TYPE"],
               st.secrets["OPENAI_API_BASE"],
               st.secrets["OPENAI_API_KEY"]).load()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
