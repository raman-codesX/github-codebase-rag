# GitHub Codebase RAG

An AI-powered Retrieval-Augmented Generation (RAG) system that analyzes a GitHub repository and answers questions about its codebase.

##  Features

* Accepts a GitHub repository URL
* Fetches repository files using the GitHub REST API
* Reads Python, Markdown, and text files
* Splits source code and documentation into chunks
* Generates embeddings using Sentence Transformers
* Stores embeddings in a FAISS vector index
* Retrieves the most relevant code/documentation chunks
* Uses Gemini to generate answers from the retrieved context

##  How It Works

```text
GitHub Repository URL
        ↓
Extract Owner & Repository
        ↓
GitHub REST API
        ↓
Get Default Branch
        ↓
Get Repository File Tree
        ↓
Fetch .py / .md / .txt Files
        ↓
Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Search
        ↓
Retrieve Relevant Chunks
        ↓
Gemini
        ↓
Answer
```

## Tech Stack

* Python
* GitHub REST API
* LangChain Text Splitters
* Sentence Transformers
* FAISS
* Gemini API
* OpenAI Python SDK
* Requests
* python-dotenv

##  Project Structure

```text
github-codebase-rag/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

> `.env` should remain local and must not be uploaded to GitHub.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/github-codebase-rag.git
cd github-codebase-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

Create a `.env` file in the project directory:

```env
GITHUB_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_api_key
```

Do not share or upload your API keys.

### 4. Add a GitHub repository URL

Inside `main.py`, provide the repository URL:

```python
file_url = "https://github.com/username/repository.git"
```

### 5. Run the project

```bash
python main.py
```

## Example Query

```text
What is this project about?
```

The system retrieves relevant information from the repository and sends that context to Gemini to generate the answer.

## Security

API keys and tokens are stored in environment variables.

The `.env` file should be included in `.gitignore`:

```text
.env
```

Never upload your GitHub token or Gemini API key to the repository.

##  Current Limitations

* Currently supports `.py`, `.md`, and `.txt` files
* Uses basic semantic similarity with FAISS
* File-source metadata is not yet preserved for each chunk
* Large repositories may require additional handling for API limits and processing time

##  Future Improvements

* Add file and chunk metadata
* Show source files with every answer
* Improve code-aware chunking
* Add hybrid search
* Add reranking
* Add a Streamlit interface
* Improve handling of large repositories
* Add codebase architecture and dependency analysis

##  Learning Goal

This project was built to understand the core components of a RAG system by connecting the retrieval pipeline manually rather than relying on a complete RAG framework.

---

**Built with Python, FAISS, Sentence Transformers, GitHub REST API, and Gemini.**
