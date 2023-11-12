import chromadb
import streamlit as st
import openai
from chromadb.utils import embedding_functions

def get_openrouter_key():
    if 'openrouter_key' not in st.session_state or st.session_state.openrouter_key is None:
        with open('openrouter.txt', 'r') as orkey:
            openrouterkey = orkey.read()
        st.session_state.openrouter_key = openrouterkey
    return st.session_state.openrouter_key

def get_current_conversation():
    if 'current_conversation' not in st.session_state or st.session_state.current_conversation is None:
        st.session_state.current_conversation = [
            {"role": "system", "content": "You are a virtual knowledge agent who is provided snippets of data from various files. You attempt to fulfill queries based on provided context whenever possible."}
        ]
    return st.session_state.current_conversation

def get_chroma_connection():
    if 'client_connection' not in st.session_state or st.session_state.client_connection is None:
        st.session_state.client_connection = chromadb.HttpClient(host='192.168.1.69', port=8000)
    return st.session_state.client_connection

def get_working_collection():
    if 'working_collection' not in st.session_state or st.session_state.working_collection is None:
        client = get_chroma_connection()
        ef = get_embedding_function()
        st.session_state.working_collection = client.get_or_create_collection(name="reporeader", embedding_function=ef)
    return st.session_state.working_collection

def get_openai_key():
    if 'openai_key' not in st.session_state or st.session_state.openai_key is None:
        with open('openai.txt', 'r') as oaikey:
            openaikey = oaikey.read()
        st.session_state.openai_key = openaikey
    return st.session_state.openai_key

def get_embedding_function():
    if 'embedding_function' not in st.session_state or st.session_state.embedding_function is None:
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key_path = "openai.txt"
        openaikey = get_openai_key()
        st.session_state.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openaikey,
            model_name="text-embedding-ada-002"
        )
    return st.session_state.embedding_function