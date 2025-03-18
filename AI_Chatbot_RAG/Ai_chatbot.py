import os
import warnings
import logging
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import docx2txt
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Suppress unnecessary warnings and configure logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

# Set up the Streamlit interface
st.title("Personal Chatbot")
st.write("Upload documents and ask questions based on their contents.")

# Initialize session state for chat messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Allow file upload for multiple document types
uploaded_files = st.file_uploader(
    "Upload documents (PDF, CSV, DOCX, TXT)",
    type=["pdf", "csv", "docx", "txt"],
    accept_multiple_files=True
)

@st.cache_resource
def build_vectorstore(uploaded_files):
    all_documents = []
    csv_row_counts = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            file_content = uploaded_file.read()

            # Create a temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # Determine the appropriate loader based on file type
                if file_extension == "pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension == "csv":
                    # Read CSV files using Pandas
                    df = pd.read_csv(temp_file_path)
                    csv_content = "\n".join([",".join(map(str, row)) for row in df.values])
                    documents = [Document(page_content=csv_content, metadata={"source": uploaded_file.name})]
                    csv_row_counts[uploaded_file.name] = len(df)
                elif file_extension == "docx":
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == "txt":
                    loader = TextLoader(temp_file_path)
                else:
                    st.error("Unsupported file format")
                    continue

                # Load file content with metadata
                if file_extension != "csv":
                    documents = loader.load()
                    for doc in documents:
                        doc.metadata = {"source": uploaded_file.name}
                all_documents.extend(documents)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue

            # Delete the temporary file after processing
            os.remove(temp_file_path)

        # Create and return a vector store from the loaded documents
        vectorstore = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
        ).from_documents(all_documents)

        return vectorstore.vectorstore, csv_row_counts
    return None, {}

# User input for querying the chatbot
user_input = st.chat_input("Enter your question")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Define the system prompt template
    system_prompt = ChatPromptTemplate.from_template("""
    You are a knowledgeable assistant, providing clear and concise answers. Answer the following question directly: {user_input}.
    Do not include introductory phrases or context references. And when i ask question from csv file than give me a answer in well
    formatted tabular way.
    """)

    model_name = "Llama3-8B-8192"
    chat_instance = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
    )

    try:
        vectorstore, csv_row_counts = build_vectorstore(uploaded_files)
        if vectorstore is None:
            st.error("Failed to create vectorstore")
            st.stop()

        # Check if the user is asking about the number of rows in a CSV file
        if "csv" in [file.name.split(".")[-1].lower() for file in uploaded_files] and "rows" in user_input.lower():
            response_text = f"The CSV file has {csv_row_counts[next(file.name for file in uploaded_files if file.name.endswith('.csv'))]} rows."
        else:
            # Set up the QA retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=chat_instance,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            response_data = qa_chain({"query": user_input})
            response_text = response_data["result"]

        st.chat_message('assistant').markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.error(f"An error occurred: {e}")
