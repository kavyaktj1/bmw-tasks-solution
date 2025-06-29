# RAG based GenAI Chatbot using Streamlit for the interface, CTransformers for the LLM, and FAISS for vector storage.
# Using Langchain for building the RAG pipeline (document loading, text splitting, embeddings, vector store, and conversational chain).

'''
python -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run app.py
'''

import streamlit as st  # for creating the web-based UI.
from streamlit_chat import message  # for rendering chat messages.
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os

# Load CSV file and convert to documents
def load_documents():
    csv_path = 'Parts.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    # Read CSV with proper delimiter and encoding
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    
    # Combine relevant columns into a single text string for each row
    documents = []
    for _, row in df.iterrows():
        # Combine all non-empty columns into a single text string
        text = ' '.join([str(val) for val in row if pd.notna(val) and val != ''])
        documents.append(Document(page_content=text, metadata={'row_id': row['ID']}))
    return documents

# Split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Create embeddings that capture context - using sentence transformer for embedding
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    return embeddings

# Create vector store (FAISS) to store the embeddings
def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

# Initialize the Mistral-7B language model for generating responses
def create_llms_model():
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm

# Initialize Streamlit app
st.title("Personalized CSV ChatBot")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)

# Load documents
try:
    documents = load_documents()
    print("-----documents loaded----")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Split text into chunks
text_chunks = split_text_into_chunks(documents)
print("-----splitted text into chunks----")

# Create embeddings
embeddings = create_embeddings()
print("-----embeddings created----")

# Create vector store
vector_store = create_vector_store(text_chunks, embeddings)
print("-----embeddings loaded to vector store----")

# Create LLM model
llm = create_llms_model()
print("-----llm model defined----")

# Initialize Streamlit's session state to store conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# List of chatbot responses to display
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Ask me anything you want to know about in the CSV data🤗"]

# List of user queries to display
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Create memory buffer for conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# Define chat function
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Display chat history and create input form
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="How can I help you?", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Render conversation history
if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")