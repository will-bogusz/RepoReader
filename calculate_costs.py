import os
import tiktoken

COST_PER_TOKEN = 0.0001 / 1000  # $0.0001 per 1K tokens
MODEL_NAME = 'gpt-3.5-turbo'

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
                with open(os.path.join(base_path, file_path), 'r') as f:
                    documents_content.append(f.read())
                    file_count += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    else:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        documents_content.append(f.read())
                        file_count += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    if documents_content:
        total_cost = total_cost_for_documents(documents_content)
        return total_cost, file_count
    else:
        return None, 0
