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
import sqlite3
import plotly.express as px
import pandas as pd

index_name = "docuqa"
dbname = "docuqa.db"


def main():
    # Set page config
    set_page_config()
    # Load llm
    llm = load_llm()
    # Initialize the vector store
    initialize_vector_store()
    # Initialize document count store
    initialize_counter_store()
    # Display introduction
    display_introduction()
    # Get uploaded document
    uploaded_file = st.file_uploader("Choose a document file", type=["pdf"])
    if uploaded_file:
        vector_store = vectorize_and_save(uploaded_file)
        # query
        user_query(vector_store, llm)
    # Display Usage stats
    display_stats()


def display_stats():
    # Retrieve the document and query counts
    document_count, query_count = retrieve_document_and_query_count()

    # Create a pandas DataFrame with the counter data
    data = {'Counter Type': ['Processed Documents', 'Queries Executed'],
            'Count': [document_count, query_count]}
    df = pd.DataFrame(data)

    # Create a bar chart of the counter data using Plotly Express
    fig = px.bar(df, x='Counter Type', y='Count', color='Counter Type',
                 height=400)

    # Add the count values to the x-axis labels
    fig.update_layout(xaxis_tickangle=-0,
                      xaxis_tickfont_size=12,
                      xaxis=dict(tickmode='array',
                                 tickvals=df['Counter Type'],
                                 ticktext=[f"{c:,d}" for c in df['Count']],
                                 title=''),
                      yaxis=dict(range=[0, max(df['Count'])]))

    # fig.update_traces(marker_color=['blue', 'light blue'])
    # Modify the chart size and margins
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        width=500,
        height=300,
    )

    # Display the chart and heading in the Streamlit app
    st.write("*App Usage Stats*")
    st.plotly_chart(fig)


def set_page_config():
    st.set_page_config(page_title="DocuQA", layout="wide")


@st.cache_resource
def initialize_vector_store():
    # initialize pinecone
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENV"]
    )


def initialize_counter_store():
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS doc_counters
            (id INTEGER PRIMARY KEY, count INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS query_counters
                (id INTEGER PRIMARY KEY, count INTEGER)''')
    conn.commit()
    conn.close()


@st.cache_resource
def load_llm():
    os.environ["OPENAI_API_TYPE"] = st.secrets["OPENAI_API_TYPE"]
    os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    return AzureOpenAI(temperature=0.9, deployment_name="text-davinci-003-dev1", model_name="text-davinci-003")


def display_introduction():
    st.header("DocuQA")
    # Intro
    st.markdown("DocuQA is the ultimate tool for anyone who needs to work with "
                "long PDF documents. Whether you're a busy professional trying to "
                "extract insights from a report or a student looking for specific information "
                "in a textbook, this powerful web app has the tools you need to get the "
                "job done quickly and efficiently. With its intuitive interface, DocuQA is "
                "the perfect solution for conducting question-answering tasks on long documents."
                )


def retrieve_document_and_query_count():
    # Retrieve total document count from the database
    document_count = 0
    query_count = 0
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('SELECT count FROM doc_counters WHERE id = 1')
    count = c.fetchone()
    if count is not None:
        document_count = count[0]

    c.execute('SELECT count FROM query_counters WHERE id = 1')
    count = c.fetchone()
    if count is not None:
        query_count = count[0]
    conn.close()
    return document_count, query_count


@st.cache_resource
def vectorize_and_save(uploaded_file):
    # Index
    index = pinecone.Index(index_name)
    # Save the uploaded file in a temp directory and load it
    document = save_and_load_document(uploaded_file)
    # split the documents into chunks
    texts = split_into_chunks(document)
    # Remove the document first
    remove_doc(index, uploaded_file)
    # Add document counter
    add_document_counter()
    return vectorize(texts, uploaded_file)


def add_document_counter():
    # Increment the document counter in the database
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('SELECT count FROM doc_counters WHERE id = 1')
    count = c.fetchone()
    if count is None:
        c.execute('INSERT INTO doc_counters (id, count) VALUES (1, ?)', (1,))
    else:
        c.execute('UPDATE doc_counters SET count = ? WHERE id = 1', (count[0] + 1,))
    conn.commit()
    conn.close()


def add_query_counter():
    # Increment the query counter in the database
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute('SELECT count FROM query_counters WHERE id = 1')
    count = c.fetchone()
    if count is None:
        c.execute('INSERT INTO query_counters (id, count) VALUES (1, ?)', (1,))
    else:
        c.execute('UPDATE query_counters SET count = ? WHERE id = 1', (count[0] + 1,))
    conn.commit()
    conn.close()


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


def split_into_chunks(document):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    return texts


def load_file(filepath):
    # Load the file
    loader = PyPDFLoader(filepath)
    document = loader.load()
    return document


def save_and_load_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(filepath)
    return loader.load()


def user_query(vector_store, llm):
    # Retrieve query results
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=vector_store.as_retriever())
    query = st.text_area(label="Ask a question", placeholder="Your question..",
                         key="text_input", value="")
    if query:
        # Add the query counter
        add_query_counter()
        # Run the query
        st.write(qa.run(query))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
