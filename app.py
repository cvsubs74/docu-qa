import os
import tempfile

import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import AzureOpenAI

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def main():
    # Display introduction
    display_introduction()
    # Load llm
    llm = load_llm()
    # Get uploaded document
    save_and_load_file(llm)
    # Vectorize and query


def display_introduction():
    st.set_page_config(page_title="DocuQA", layout="wide")
    st.header("DocuQA")
    st.markdown("DocuQA is the ultimate tool for anyone who needs to work with "
                "long PDF documents. Whether you're a busy professional trying to "
                "extract insights from a report or a student looking for specific information "
                "in a textbook, this powerful web app has the tools you need to get the "
                "job done quickly and efficiently. With its intuitive interface and advanced "
                "natural language processing capabilities, DocuQA is the perfect solution for "
                "conducting question-answering tasks on long documents."
                )


def save_and_load_file(llm):
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf"])
    loader = None
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if os.path.exists(filepath):
            loader = PyPDFLoader(filepath)
            if loader is not None:
                document = loader.load()
                # split the documents into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(document)
                # select which embeddings we want to use
                embeddings = OpenAIEmbeddings()
                # create the vector store to use as the index
                vector_store = FAISS.from_documents(texts, embeddings)
                chat_history = []
                qa = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever(), verbose=False)
                query = "Summarize the document"
                result = qa({"question": query, "chat_history": chat_history})
                st.write(result)
                chat_history = [(query, result["answer"])]
                query = "Give me more details about windows"
                result = qa({"question": query, "chat_history": chat_history})
                st.write(result)


def load_llm():
    os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    llm = AzureOpenAI(temperature=0.9, deployment_name="text-davinci-003-dev1", model_name="text-davinci-003")
    return llm


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
