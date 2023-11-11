import streamlit as st
import os

def traverse_directories(root_path):
    tree = {}
    base_length = len(root_path.split(os.sep))
    for root, dirs, files in os.walk(root_path):
        path = root.split(os.sep)[base_length:]
        parent = tree
        for dir in path:
            parent = parent.setdefault(dir, {})
        parent['_files'] = files
    return tree

def render_tree(tree, path="", level=0):
    indent = "→ " * level
    folder_emoji = "📁 "
    file_emoji = "📄 "
    for branch in tree:
        if branch == '_files':
            if tree[branch]:
                for file in tree[branch]:
                    st.checkbox(indent + file_emoji + file, key=f"file:{path + file}")
        else:
            if st.checkbox(indent + folder_emoji + branch, key=path + branch):
                render_tree(tree[branch], path + branch + os.sep, level + 1)


def get_selected_files():
    selected_files = []
    for key, value in st.session_state.items():
        if value and "file:" in key:
            file_path = key.split("file:", 1)[1]
            selected_files.append(file_path)
    return selected_files


def directory_explorer_app():
    directory_tree = traverse_directories("C:\\Users\\wbogu\\Temp\\")
    render_tree(directory_tree)
    
    # Display the "Done" button
    if st.button("Done"):
        selected_files = get_selected_files()
        st.session_state['selected_files'] = selected_files
        st.session_state['explore_directory'] = False
        # Optionally, display the selected files or handle them as needed
        st.write("Selected Files:", selected_files)
        st.experimental_rerun()


# This part of the code will now only be executed when the directory explorer is active
if 'explore_directory' in st.session_state and st.session_state['explore_directory']:
    directory_explorer_app()