import subprocess
import os

def clone_repo(git_url, local_path):
    # Extract the repo name from the URL to use as the directory name
    repo_name = git_url.strip('/').rstrip('.git').split('/')[-1]

    # Complete local path where the repo will be cloned
    full_local_path = os.path.join(local_path, repo_name)
    full_local_path = os.path.normpath(full_local_path)

    # Check if the directory already exists
    if os.path.isdir(full_local_path):
        print(f"Repository already exists locally at '{full_local_path}'")
        return

    # Ensure the base local_path exists (not the full_local_path since git will create the repo_name dir)
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Execute git clone command
    try:
        subprocess.run(['git', 'clone', git_url, full_local_path], check=True)
        print(f'Repository cloned successfully to {full_local_path}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while cloning the repository: {e}')
    except Exception as e:
        print(f'Something went wrong: {e}')

# Input URL and local path
repo_url = input('Enter the GitHub repository URL: ')

# Define the base local directory where you have write permissions, without the repo name
# Example: "C:/Users/wbogu/Temp" or any other desired base directory
base_local_dir = "C:/Users/wbogu/Temp"

# Clone the repository
clone_repo(repo_url, base_local_dir)