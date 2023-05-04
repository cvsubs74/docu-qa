import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

index_name = "docuqa"


def main():
    # Display introduction
    display_introduction()
    # Load llm
    llm = load_llm()
    # Initialize the vector store
    initialize_vector_store()
    # Get uploaded document
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf"])
    if uploaded_file:
        vector_store = vectorize(uploaded_file)
        # query
        user_query(vector_store, llm)


@st.cache_resource
def initialize_vector_store():
    # initialize pinecone
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"]
    )


@st.cache_resource
def load_llm():
    print(">>load_llm")
    os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    return AzureOpenAI(temperature=0.9, deployment_name="text-davinci-003-dev1", model_name="text-davinci-003")


def display_introduction():
    print(">>display_introduction")
    st.set_page_config(page_title="DocuQA", layout="wide")
    st.header("DocuQA")
    st.markdown("DocuQA is the ultimate tool for anyone who needs to work with "
                "long PDF documents. Whether you're a busy professional trying to "
                "extract insights from a report or a student looking for specific information "
                "in a textbook, this powerful web app has the tools you need to get the "
                "job done quickly and efficiently. With its intuitive interface, DocuQA is "
                "the perfect solution for conducting question-answering tasks on long documents."
                )


@st.cache_resource
def vectorize(uploaded_file):
    print(">>vectorize")
    index = pinecone.Index("docuqa")
    vector_store = None
    temp_dir = tempfile.TemporaryDirectory()
    filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(filepath)
        if loader is not None:
            document = loader.load()
            # split the documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(document)
            # select which embeddings we want to use
            embeddings = OpenAIEmbeddings(chunk_size=1)
            # create the vector store to use as the index
            index.delete(delete_all=True, namespace=uploaded_file.name)
            vector_store = Pinecone.from_texts(
                [t.page_content for t in texts], embeddings,
                index_name=index_name, namespace=uploaded_file.name)
    return vector_store


def user_query(vector_store, llm):
    print(">>user_query")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=vector_store.as_retriever())
    query = st.text_area(label="Ask a question", placeholder="Your question..",
                         key="text_input", value="")
    if query:
        st.write(qa.run(query))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
