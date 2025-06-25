#RAG based GenAI Chatbot using Streamlit for the interface, CTransformers for the LLM, and FAISS for vector storage.
#using Langchain for building the RAG pipeline (document loading, text splitting, embeddings, vector store, and conversational chain).

'''
python -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run app.py
'''

import streamlit as st # for creating the web-based UI.
from streamlit_chat import message #for rendering chat messages.
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# load documents like pdf to query
def load_documents():
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# split text into chunks 
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# create embeddings that capture context - using sentence transformer for embedding
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    return embeddings

# create vector store(FAISS) to store the embeddings.
# helps in efficient similarity search in the vector store.
def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

# Initializing the Mistral-7B language model for generating responses.
# Using CTransformers for loading and running a quantized language model (Mistral-7B).
def create_llms_model():
    llm = CTransformers(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm

# we can also use  transformers framework from Hugging Face and do quantisation by ourself using BitsandBytes. 
# Here tokenizer and model is loaded separately. 
def create_llms_model_hf():
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # More powerful, GPU-friendly model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    return model, tokenizer, device

# Initialize Streamlit app
st.title("Personalized PDF ChatBot")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)

# loading of documents
documents = load_documents()
print("-----documents loaded----")

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

# Initializes Streamlit's session state to store conversation history
# List of (query, response) tuples for the conversational chain
if 'history' not in st.session_state:
    st.session_state['history'] = []

# List of chatbot responses to display.
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Ask me anything you want to know about in PDF uploaded🤗"]

#List of user queries to display.
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Creates a memory buffer to store and retrieve the chat history for context-aware responses.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Combines the language model, vector store retriever, and memory into a conversational chain.
# chain_type='stuff' = Combines retrieved documents into a single prompt for the LLM.
# memory: Maintains conversation context.
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

# Define chat function - Processes a user query and returns the chatbot’s response.
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Display chat history - chat interface - Creates the input form for users to submit queries.
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="How can I help you?", key='input')
        submit_button = st.form_submit_button(label='Send')
    
    #When a query is submitted, it calls conversation_chat, stores the query in past, and the response in generated.
    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

#Renders the conversation history in the UI.
if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
