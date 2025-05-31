#Conversational Q&A RAG with PDF uploads including chat history

import os
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchian.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload pdfs and chat with their content")

api_key=st.text_input("Enter your GROQ API key", type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="distil-whisper-large-v3-en")

    session_id=st.text_input("Session ID", value="default_session")

    