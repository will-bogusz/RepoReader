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

from repo_explorer import directory_explorer_app
from calculate_costs import calculate_cost_from_selection

# method for downloading a github repository to a local file system
def clone_repo(git_url):
    local_path = "C:/Users/wbogu/Temp/Repositories"
    repo_name = git_url.strip('/').rstrip('.git').split('/')[-1]
    full_local_path = os.path.join(local_path, repo_name)

    if os.path.isdir(full_local_path):
        print(f"Repository already exists locally at '{full_local_path}'")
        return full_local_path

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    try:
        subprocess.run(['git', 'clone', git_url, full_local_path], check=True)
        print(f'Repository cloned successfully to {full_local_path}')

        # Remove .git, .gitignore, and .github files/folders after cloning
        git_dir = os.path.join(full_local_path, '.git')
        gitignore_file = os.path.join(full_local_path, '.gitignore')
        github_dir =os.path.join(full_local_path, '.github')

        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)
        if os.path.exists(github_dir):
            shutil.rmtree(github_dir)
        if os.path.exists(gitignore_file):
            os.remove(gitignore_file)

        return full_local_path
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while cloning the repository: {e}')
        return ""
    except Exception as e:
        print(f'Something went wrong: {e}')
        return ""

def get_model_response(query):
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key_path = "openrouter.txt"

    messages = get_current_conversation()

    prompt = {"role": "user", "content": query}

    messages.append(prompt)

    response = openai.ChatCompletion.create(
        #model = "",
        model="openai/gpt-3.5-turbo-1106",
        messages=messages,
        headers={
            "HTTP-Referer": "http://bogusz.co",
        },
    )

    response_content = response.choices[0].message

    messages.append(response_content)

    openai.api_base = "https://api.openai.com/v1"
    openai.api_key_path = "openai.txt"

    return response_content["content"]

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

def inject_context(query):
    collection = get_working_collection()

    vector = embed_text(query)

    # modify this later to include the metadata so that we can provide citations
    results = collection.query(
        query_embeddings=vector,
        n_results=15,
        include=["documents", "metadatas"]
    )

    #print(results)

    # get the text from the results, create a query header to assist model in differentiating sources of data
    structured_context = f"""
        **Query for Analysis:**
        {query}

        **Provided Context:**
        The following sections contain mixed types of data from various sources for context. Each section is clearly marked with its source origin and type. The content includes both plain text and formatted code blocks, as indicated.

        **Contexts:**

        """

    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        structured_context += f"{i+1}. **[Source: {meta['source']}, Type: {meta['content_type']}, Language: {meta['language']}]**\n```\n{doc}\n```\n\n"


    return structured_context


def begin_conversation():
    user_question = st.text_input('Ask a question about your uploaded data:')

    # will want to find a better way to display the conversation with a history
    if user_question:
        query = inject_context(user_question)
        response = get_model_response(query)

        conversation = get_current_conversation()

        st.write(conversation)

    
def embed_document(doc):
    val = 'placeholder'
    # based on file type decide a method for chunking the text

    # do any data cleaning of the chunks, metadata, etc.

    # generate embedding for each chunk using 'embed_text()'

    # add the embedding to chromadb

def upload_documents():
    valid_data_documents = ["doc", "txt", "md", "pdf", "log", "py", "js"]
    document = st.file_uploader("Upload your data", type=valid_data_documents)
    embed = st.button('Load Documents')
    if document is not None and embed:
        embed_document(document)

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
