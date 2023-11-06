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

# method for downloading a github repository to a local file system
def clone_repo(git_url):

    local_path = "C:/Users/wbogu/Temp"
    # Extract the repo name from the URL to use as the directory name
    repo_name = git_url.strip('/').rstrip('.git').split('/')[-1]

    # Complete local path where the repo will be cloned
    full_local_path = local_path + "/" + repo_name

    # Check if the directory already exists
    if os.path.isdir(full_local_path):
        print(f"Repository already exists locally at '{full_local_path}'")
        return full_local_path

    # Ensure the base local_path exists (not the full_local_path since git will create the repo_name dir)
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Execute git clone command
    try:
        subprocess.run(['git', 'clone', git_url, full_local_path], check=True)
        print(f'Repository cloned successfully to {full_local_path}')
        return full_local_path
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while cloning the repository: {e}')
        return ""
    except Exception as e:
        print(f'Something went wrong: {e}')
        return ""


def store_documents(docs):
    with open('openai.txt', 'r') as oaikey:
        openaikey = oaikey.read()

    openai.api_key = openaikey

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openaikey,
        model_name="text-embedding-ada-002"
    )

    client = chromadb.HttpClient(host='192.168.1.69', port=8000)

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=150, chunk_overlap=0
    )

    # clean up the data before it is added, we will be computing embeddings before adding them to the db
    # metadata will also be separated for entry

    # if it was a valid source file, clean up the metadata entry

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
            if not None:
                curr_meta['language'] = lang.value
        metadata.append(stringify_dictionary(curr_meta))

        # create the embedding for the chunk
        embed = openai.Embedding.create(
          model="text-embedding-ada-002",
          input=doc.page_content,
          encoding_format="float"
        )

        #print(embed)

        final_embedding = embed.data[0]['embedding']
        #print(final_embedding)
        embeddings.append(final_embedding)


    collection = client.get_or_create_collection(name="reporeader", embedding_function=openai_ef)

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

def main():

    st.set_page_config(page_title="Ask your Codebase")
    st.header("Ask your Codebase")

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
        loader = GenericLoader.from_filesystem(
            repo_final_paths[0],
            glob="main*",
            suffixes=[".py", ".js"],
            parser=LanguageParser()
        )
        docs = loader.load()

        #print(len(docs))
        #print(type(docs[0].metadata['language']))

        #print(docs[0])

        store_documents(docs)

if __name__ == '__main__':
    main()
