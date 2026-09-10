
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")

#paste your github url
file_url = "https://github.com/raman-codesX/mini-project.git"

parts = file_url.rstrip("/").split("/")

owner = parts[-2]
repo = parts[-1].removesuffix(".git")

headers = {
    "Authorization": f"Bearer {github_token}",
    "Accept": "application/vnd.github+json"
}



repo_api = f"https://api.github.com/repos/{owner}/{repo}"

response = requests.get(
    repo_api,
    headers=headers
)
repo_data = response.json()
print("Status Code:", response.status_code)
if response.status_code != 200:
    print("Github API Error!")
    print(repo_data)
    exit()

branch = repo_data["default_branch"]

api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

response = requests.get(
    api_url,
    headers=headers
)
data = response.json()
print("Tree Status Code:", response.status_code)
if response.status_code != 200:
    print("Tree Api Error!")
    print(data)
    exit()
all_text = []

for item in data["tree"]:
    if item["type"] == "blob":
        path = item["path"]

        if path.endswith((".py", ".md", ".txt", )):
            File_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            response = requests.get(
                File_url,
                headers=headers
            )
            file_data = response.json()
            print("Path :", path)
            print("status:", response.status_code)
            if response.status_code != 200:
                print("Could not read!:", path)
                continue

            if "content" not in file_data:
                continue


            content = base64.b64decode(
                file_data["content"]
            ).decode("utf-8")

            all_text.append(content)



#Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
    
)

chunks = splitter.split_text("\n".join(all_text))
print("Chunks : ", len(chunks))

#Embedding
print("---Start Embedding---")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(chunks)
print("---Complete Embedding---")

#faiss
dimension = embedding.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding)

query = "what is that or useful or not?"
query_embedding = model.encode([query])


distance, indices = index.search(query_embedding, 3)

retrieved_chunks = []
for i in indices[0]:
    retrieved_chunks.append(chunks[i])

context = '\n\n'.join(retrieved_chunks)

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )

response = client.chat.completions.create(
    model="gemini-3.6-flash",
      messages=[
        {"role": "system", "content": "Answer the question using only the provided context"},
        {"role": "user", "content": f"""
context:
{context}

Query:
{query}
"""}
      ]
        
)

print(response.choices[0].message.content)
print("Token exists:", github_token is not None)
print("Token length:", len(github_token) if github_token else 0)
