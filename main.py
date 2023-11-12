import os
import streamlit as st
from dotenv import load_dotenv
import openai
import shutil
import time
import stat


import streamlit as st
from document_handler import prompt_for_urls, upload_documents, store_documents, calculate_cost_from_selection
from chat_handler import get_model_response, begin_conversation
from repo_explorer import directory_explorer_app


def format_file_hierarchy(selected_files):
    file_hierarchy = {}
    for file_path in selected_files:
        file_path = file_path[len("C:\\Users\\wbogu\\Temp\\"):]
        parts = file_path.split(os.sep)
        repo_name = parts[0]
        sub_path = os.sep.join(parts[1:-1])
        filename = parts[-1]

        if repo_name not in file_hierarchy:
            file_hierarchy[repo_name] = {}
        if sub_path not in file_hierarchy[repo_name]:
            file_hierarchy[repo_name][sub_path] = []

        file_hierarchy[repo_name][sub_path].append(filename)
    return file_hierarchy

def display_selected_files(selected_files, title):
    if selected_files:
        file_hierarchy = format_file_hierarchy(selected_files)

        st.sidebar.markdown(f"**{title}**")
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
        st.sidebar.write(f"No files currently selected for {title.lower()}")


def main():
    st.set_page_config(page_title="Ask your Codebase")
    st.header("Ask your Codebase")

    selected_files_embedding = st.session_state.get('selected_files', [])
    display_selected_files(selected_files_embedding, "Selected For Embedding")

    # Display 'Selected For Context'
    selected_files_context = st.session_state.get('selected_context_files', [])
    display_selected_files(selected_files_context, "Selected For Context")


    select_for_embedding = st.button("Select For Embedding")
    select_for_context = st.button("Select For Context")

    calculate_cost_btn = st.button("Calculate Cost")
    embed_documents_btn = st.button("Embed Documents")

    if select_for_embedding:
        st.session_state['explore_directory'] = True
        st.session_state['directory_target'] = 'selected_files'

    if select_for_context:
        st.session_state['explore_directory'] = True
        st.session_state['directory_target'] = 'selected_context_files'

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
            store_documents(selected_files)
        else:
            total_cost, file_count = calculate_cost_from_selection(selected_files, base)
            if total_cost is not None and file_count > 0:
                st.session_state['confirm_embed_all'] = True
                st.success(f"Total cost to embed all {file_count} documents: ~${total_cost:.3f}")

    if 'confirm_embed_all' in st.session_state and st.session_state['confirm_embed_all']:
        if st.button("Confirm"):
            st.session_state['confirm_embed_all'] = False
            store_documents(selected_files)
            st.rerun()
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

    if 'explore_directory' in st.session_state and st.session_state['explore_directory']:
        directory_target = st.session_state.get('directory_target', 'selected_files')
        directory_explorer_app(directory_target)

if __name__ == '__main__':
    main()
