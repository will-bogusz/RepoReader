import openai
import streamlit as st
from utils import get_current_conversation, get_working_collection, get_cohere_client, call_with_timeout
import cohere

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

def get_chunk_classification(query, metadata):
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key_path = "openrouter.txt"

    augmented_prompt = f"""Translate the following code snippet into a user-friendly description. The description should be structured to facilitate keyword searches commonly used by someone trying to understand or locate specific functionalities within a codebase. Consider the following objectives:

        1. Clearly identify and name each significant component (function, method, class) in the code and make direct reference to the filename and repository (source).
        2. Explain the purpose and functionality of these components in simple terms.
        3. Use common programming terminology that users might employ in their queries, such as "function", "method", "class", "return value", "parameter", "loop", etc.
        4. Highlight any specific tasks or operations the code performs, which users might search for, like "sorting a list", "calculating a sum", "handling user input", etc.
        5. Your translation should reflect the scale of the snippet, i.e. large snippets have more description, but a single line snippet evokes a single line translation
        6. Explain how the operations might work together and describe the possible role or purpose of this snippet within a larger codebase.

        Remember, the goal is to make the code snippet's functionality and role within a larger codebase easily discoverable through search queries. Do not comment on irrelevant features.

        Metadata:
        {metadata}

        Snippet:
        {query}
    """

    messages = [
        {"role": "system", "content": "You are a virtual knowledge agent who is provided snippets of data from various files. You attempt to fulfill queries based on provided context."},
        {"role": "user", "content": augmented_prompt}
    ]

    response = openai.ChatCompletion.create(
        #model = "",
        model="openai/gpt-3.5-turbo-1106",
        messages=messages,
        headers={
            "HTTP-Referer": "http://bogusz.co",
        },
    )

    response_content = response.choices[0].message

    openai.api_base = "https://api.openai.com/v1"
    openai.api_key_path = "openai.txt"

    return response_content["content"]

def inject_context(query):
    collection = get_working_collection()

    vector, error = call_with_timeout(embed_text, [query], 30)
    if error:
        print(f"Error or timeout on first try embedding: {error}. Retrying...")
        vector, error = call_with_timeout(embed_text, [query], 30)
        if error:
            print(f"Error or timeout on second try embedding query: {error}.")
            raise Exception("Unable to vectorize query, failed embedding")

    results = collection.query(
        query_embeddings=vector,
        n_results=100,
        include=["documents", "metadatas"]
    )

    # Prepare documents for reranking
    documents_for_reranking = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        content = meta.get('translation') if meta.get('translation') and len(meta['translation']) > 5 else doc
        documents_for_reranking.append({"text": content})

    co = get_cohere_client()

    # Complete rerank call
    reranked_results = co.rerank(
        query=query,
        documents=documents_for_reranking,
        model="rerank-english-v2.0",
        top_n=10
    )

    structured_context = f"""
        **Query for Analysis:**
        {query}

        **Provided Context:**
        The following sections contain mixed types of data from various sources for context. Each section is clearly marked with its source origin and type. The content includes both plain text and formatted code blocks, as indicated.

        **Contexts:**

        """

    for rank in reranked_results.results:
        doc_index = rank.index
        doc = documents_for_reranking[doc_index]["text"]
        meta = results['metadatas'][0][doc_index]
        structured_context += f"{rank.index+1}. **[Source: {meta['source']}, Type: {meta['type']}, Language: {meta['language']}, Filename: {meta['filename']}]**\n```\n{doc}\n```\n\n"

    return structured_context



def begin_conversation():
    user_question = st.text_input('Ask a question about your uploaded data:')

    # will want to find a better way to display the conversation with a history
    if user_question:
        query = inject_context(user_question)
        response = get_model_response(query)

        conversation = get_current_conversation()

        st.write(conversation)