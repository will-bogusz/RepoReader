import openai
import streamlit as st
from utils import get_current_conversation, get_working_collection
from document_handler import embed_text

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

def get_chunk_classification(query):
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key_path = "openrouter.txt"

    augmented_prompt = f"""Translate the following code snippet into a user-friendly description. The description should be structured to facilitate keyword searches commonly used by someone trying to understand or locate specific functionalities within a codebase. Consider the following objectives:

        1. Clearly identify and name each significant component (function, method, class) in the code.
        2. Explain the purpose and functionality of these components in simple terms.
        3. Use common programming terminology that users might employ in their queries, such as "function", "method", "class", "return value", "parameter", "loop", etc.
        4. Highlight any specific tasks or operations the code performs, which users might search for, like "sorting a list", "calculating a sum", "handling user input", etc.
        5. If possible, relate the code to typical programming problems or scenarios where such a code snippet would be relevant.

        Remember, the goal is to make the code snippet's functionality and role within a larger codebase easily discoverable through search queries. 

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