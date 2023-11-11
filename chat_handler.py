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