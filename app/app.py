import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

# Load the OpenAI API and Groq API
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Defining st features
st.title("PDF QA Bot (Better Best Software Solutions)")

# Configuring the llm
llm = ChatGroq(model= "llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context only.\
    Please provide most accurate result based on the questions.\
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./input_files")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Save uploaded files to 'input_files' directory
if uploaded_files:
    os.makedirs('input_files', exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join('input_files', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
    st.write("Files uploaded successfully!")

prompt1 = st.text_input('Enter your question for your document')

if st.button("Embed Document"):
    vector_embeddings()
    st.write("Embedding Completed, Vector DB is ready with data")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
