import os
import tiktoken
import subprocess
import shutil
import stat
import uuid
import streamlit as st
import chromadb
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from chromadb.utils import embedding_functions
import openai
import time
from utils import get_working_collection
import chardet

COST_PER_TOKEN = 0.0001 / 1000  # $0.0001 per 1K tokens
MODEL_NAME = 'gpt-3.5-turbo'

def get_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        return chardet.detect(file.read())['encoding']

def set_permissions_recursive(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IWRITE)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IWRITE)

def is_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.read()
        return True
    except UnicodeDecodeError:
        return False

def clean_cloned_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.normpath(os.path.join(root, file))
            if not is_text_file(file_path):
                print(f"Removing file: {file_path}")
                os.remove(file_path)

def clone_repo(git_url):
    local_path = "C:/Users/wbogu/Temp/Repositories"
    repo_name = git_url.strip('/').rstrip('.git').split('/')[-1]
    full_local_path = os.path.normpath(os.path.join(local_path, repo_name))

    if os.path.isdir(full_local_path):
        print(f"Repository already exists locally at '{full_local_path}'")
        return full_local_path

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    try:
        subprocess.run(['git', 'clone', git_url, full_local_path], check=True)
        print(f'Repository cloned successfully to {full_local_path}')

        set_permissions_recursive(full_local_path)

        # Remove .git, .gitignore, and .github files/folders after cloning
        git_dir = os.path.normpath(os.path.join(full_local_path, '.git'))
        gitignore_file = os.path.normpath(os.path.join(full_local_path, '.gitignore'))
        github_dir = os.path.normpath(os.path.join(full_local_path, '.github'))

        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)
        if os.path.exists(github_dir):
            shutil.rmtree(github_dir)
        if os.path.exists(gitignore_file):
            os.remove(gitignore_file)

        time.sleep(1)
        clean_cloned_directory(full_local_path)

        return full_local_path
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while cloning the repository: {e}')
        return ""
    except Exception as e:
        print(f'Something went wrong: {e}')
        return ""

def prompt_for_urls():
    # Define the key for our dynamic inputs in the session state
    if 'url_inputs' not in st.session_state:
        st.session_state.url_inputs = ['']

    # Define function to add a new URL input
    def add_url_input():
        st.session_state.url_inputs.append('')

    # Define function to remove the last URL input
    def remove_url_input():
        if len(st.session_state.url_inputs) > 1:
            st.session_state.url_inputs.pop()

    # Display the URL inputs dynamically
    for i, _ in enumerate(st.session_state.url_inputs):
        st.session_state.url_inputs[i] = st.text_input(f"URL {i+1}", value=st.session_state.url_inputs[i], key=f"url_{i}")

    # Add and remove URL buttons
    col1, col2 = st.columns(2)
    with col1:
        add_button = st.button("Add URL", on_click=add_url_input)
    with col2:
        remove_button = st.button("Remove Last URL", on_click=remove_url_input)

    # Button to submit URLs
    submit = st.button('Submit')
    if submit:
        st.write("Submitted URLs:")
        repo_final_paths = []
        for url in st.session_state.url_inputs:
            if url:  # Check if the URL is not empty
                st.write(url)
                # Call clone_repo and collect the cloned repo paths
                repo_path = clone_repo(url)
                if repo_path:
                    repo_final_paths.append(repo_path)
                # Provide feedback on successful clone
                st.success(f"Cloned {url} successfully!")
            else:
                # Provide feedback for empty URL field
                st.error("Please enter a URL.")

def embed_text(text):
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key_path = "openai.txt"
    passed = False

    for j in range(5):
        try:
            res = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
            passed = True
        except openai.error.RateLimitError:
            time.sleep(2**j)
    if not passed:
        raise RuntimeError("Failed to create embeddings.")
    embedding = res['data'][0]['embedding']

    return embedding

def store_documents(docs):
    print(docs)
    return
    loader = GenericLoader.from_filesystem(
        "placeholder",
        glob="**/*",
        suffixes=[".py", ".js"],
        parser=LanguageParser()
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=150, chunk_overlap=0
    )

    # clean up the data before it is added, we will be computing embeddings before adding them to the db
    # metadata will also be separated for entry

    # if it was a valid source file, clean up the metadata entry

    path_prefix = "C:\\Users\\wbogu\\Temp\\"

    uuids = []
    metadata = []
    embeddings = []
    text = []
    for doc in docs:
        text.append(doc.page_content)
        # create a uuid identifier to tag the document in chroma
        uuids.append(str(uuid.uuid4()))
        # handle metadata
        curr_meta = doc.metadata
        # check if it contains anything
        if curr_meta:
            lang = curr_meta.get('language')
            path = curr_meta.get('source')
            print(curr_meta)
            if lang:
                curr_meta['language'] = lang.value
            if path:
                # remove the download path prefix from the source so that it can be better interpretted by the model 
                cleaned = path.replace(path_prefix, "", 1)
                print(path_prefix)
                print(cleaned)
                curr_meta['source'] = cleaned
        metadata.append(stringify_dictionary(curr_meta))

        # create the embedding for the chunk
        embed = embed_text(doc.page_content)

        #print(embed)

        final_embedding = embed.data[0]['embedding']
        #print(final_embedding)
        embeddings.append(final_embedding)


    collection = get_working_collection()

    total_embeddings = collection.count()
    print(f"Collection currently has {total_embeddings} embeddings!")


    collection.add(
        documents=text,
        embeddings=embeddings,
        metadatas=metadata,
        ids=uuids
    )

    total_embeddings = collection.count()
    print(f"Collection now has {total_embeddings} embeddings!")

# helper to clean up metadata
def stringify_dictionary(input_dict):
    return {str(key): (str(value) if not isinstance(value, str) else value) for key, value in input_dict.items()}

def upload_documents():
    valid_data_documents = ["doc", "txt", "md", "pdf", "log", "py", "js"]
    document = st.file_uploader("Upload your data", type=valid_data_documents)
    embed = st.button('Load Documents')
    if document is not None and embed:
        embed_document(document)

def count_tokens(text):
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return len(encoding.encode(text))

def calculate_cost(text):
    token_count = count_tokens(text)
    return token_count * COST_PER_TOKEN

def total_cost_for_documents(documents):
    return sum(calculate_cost(doc) for doc in documents)

def calculate_cost_from_selection(selected_files, base_path):
    documents_content = []
    file_count = 0

    if selected_files:
        for file_path in selected_files:
            try:
                encoding = get_file_encoding(full_path) or 'utf-8'
                with open(full_path, 'r', encoding=encoding) as f:
                    documents_content.append(f.read())
                    file_count += 1
            except Exception as e:
                print(f"Error reading file {full_path}: {e}")
    else:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    encoding = get_file_encoding(file_path) or 'utf-8'
                    with open(file_path, 'r', encoding=encoding) as f:
                        documents_content.append(f.read())
                        file_count += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    if documents_content:
        total_cost = total_cost_for_documents(documents_content)
        return total_cost, file_count
    else:
        return None, 0