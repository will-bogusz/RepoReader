import subprocess
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from chromadb.utils import embedding_functions
import uuid
import chromadb
from chromadb.config import Settings
import openai
import shutil
import time
import stat


import streamlit as st
from document_handler import prompt_for_urls, upload_documents, store_documents, calculate_cost_from_selection
from chat_handler import get_model_response
from repo_explorer import directory_explorer_app


def format_file_hierarchy(selected_files):
    file_hierarchy = {}
    for file_path in selected_files:
        parts = file_path.split(os.sep)
        repo_name = parts[0]
        sub_path = os.sep.join(parts[1:-1])  # Get the subdirectory path
        filename = parts[-1]

        # Initialize dictionary structure
        if repo_name not in file_hierarchy:
            file_hierarchy[repo_name] = {}
        if sub_path not in file_hierarchy[repo_name]:
            file_hierarchy[repo_name][sub_path] = []

        file_hierarchy[repo_name][sub_path].append(filename)
    return file_hierarchy


def main():
    st.set_page_config(page_title="Ask your Codebase")
    st.header("Ask your Codebase")

    st.sidebar.header("Selected Files")
    if 'selected_files' in st.session_state and st.session_state['selected_files']:
        selected_files = st.session_state.get('selected_files', [])
        file_hierarchy = format_file_hierarchy(selected_files)

        for repo_name, subdirs in file_hierarchy.items():
            st.sidebar.markdown(f"**{repo_name}**")
            for subdir, files in subdirs.items():
                if subdir:  # Check if the subdir is not empty
                    st.sidebar.markdown(f"_{subdir}_")
                st.sidebar.markdown("<ul>", unsafe_allow_html=True)
                for filename in files:
                    st.sidebar.markdown(f"<li>{filename}</li>", unsafe_allow_html=True)
                st.sidebar.markdown("</ul>", unsafe_allow_html=True)
    else:
        st.sidebar.write("No files currently selected")


    explore_directories = st.button("Explore Directories")
    calculate_cost_btn = st.button("Calculate Cost")
    embed_documents_btn = st.button("Embed Documents")

    if explore_directories:
        st.session_state['explore_directory'] = True

    base = "C:\\Users\\wbogu\\Temp\\"
    selected_files = st.session_state.get('selected_files', [])

    if calculate_cost_btn:        
        total_cost, file_count = calculate_cost_from_selection(selected_files, base)
        if total_cost is not None and file_count > 0:
            st.success(f"Total cost of embedding {file_count} files: ~${total_cost:.3f}")
        elif file_count == 0:
            st.error("No documents found for cost calculation.")
        else:
            st.error("Please explore the directory and select files first or ensure files are accessible.")

    if embed_documents_btn:
        if selected_files:
            store_repo_documents(selected_files, base)  # Assuming this function takes a list of file paths
        else:
            # Confirm embedding all documents
            total_cost, file_count = calculate_cost_from_selection(selected_files, base)
            if total_cost is not None and file_count > 0:
                st.session_state['confirm_embed_all'] = True
                st.write(f"Total cost to embed all documents: ~${total_cost:.3f}")

    if 'confirm_embed_all' in st.session_state and st.session_state['confirm_embed_all']:
        if st.button("Confirm"):
            store_repo_documents(selected_files, base_path=base)
            st.session_state['confirm_embed_all'] = False
        if st.button("Cancel"):
            st.session_state['confirm_embed_all'] = False
            st.rerun()

    if not st.session_state.get('explore_directory', False):
        show_github_ingestion = st.checkbox('Show Repository Loader')
        show_conversation_box = st.checkbox('Show Conversation Box')
        show_new_document_loader = st.checkbox('Show Document Uploader')

        if show_github_ingestion:
            prompt_for_urls()

        if show_new_document_loader:
            upload_documents()

        if show_conversation_box:
            begin_conversation()

    if st.session_state.get('explore_directory', False):
        directory_explorer_app()

if __name__ == '__main__':
    main()
