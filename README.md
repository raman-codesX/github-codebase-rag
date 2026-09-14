# CodeBase AI

A GitHub Codebase RAG application that lets users understand, search, and analyze a repository using natural-language questions.

## Features

* Analyze GitHub repositories
* Fetch `.py`, `.md`, and `.txt` files through GitHub API
* Code/text chunking
* Semantic search with FAISS
* Keyword search with BM25
* Hybrid retrieval
* CrossEncoder reranking
* Gemini-powered answers
* Streamlit chat interface
* Source file references

## Architecture

```text
GitHub Repository
       ↓
   GitHub API
       ↓
   File Loading
       ↓
     Chunking
       ↓
 ┌───────────────┐
 │               │
FAISS           BM25
 │               │
 └───────┬───────┘
         ↓
   Hybrid Search
         ↓
 CrossEncoder
   Reranking
         ↓
 Relevant Chunks
         ↓
      Gemini
         ↓
      Answer
```

## Tech Stack

| Component         | Technology            |
| ----------------- | --------------------- |
| Language          | Python                |
| UI                | Streamlit             |
| Embeddings        | Sentence Transformers |
| Vector Search     | FAISS                 |
| Keyword Search    | BM25                  |
| Reranking         | CrossEncoder          |
| LLM               | Google Gemini         |
| Repository Access | GitHub REST API       |

## How It Works

### 1. Repository Fetching

The user provides a GitHub repository URL. The application uses the GitHub API to retrieve the repository tree and supported files.

### 2. Chunking

Files are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### 3. Embeddings

Chunks are converted into embeddings using:

```text
all-MiniLM-L6-v2
```

These embeddings are stored in a FAISS index.

### 4. Retrieval

Two retrieval methods are used:

* **FAISS** — semantic similarity
* **BM25** — keyword matching

Their results are combined to create hybrid search results.

### 5. Reranking

The retrieved chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves the relevance of the final context.

### 6. Answer Generation

The top relevant chunks are passed to Gemini along with the user's question.

Gemini generates the final answer using the retrieved code as context.

## Project Structure

```text
mini-project/
│
├── main.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/raman-codesX/mini-project.git
cd mini-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token
```

Do not commit `.env` or API keys to GitHub.

## Run

Start the Streamlit application:

```bash
streamlit run main.py
```

## Example Questions

```text
What does this repository do?
```

```text
How is FAISS used in this project?
```

```text
Explain the hybrid search implementation.
```

```text
How does the CrossEncoder reranker work?
```

## Current Limitations

* Supports `.py`, `.md`, and `.txt` files
* Large repositories may take longer to process
* GitHub API limits may apply
* Gemini API usage depends on the available quota
* The repository is currently processed when a query is submitted

## Future Improvements

* Persistent vector database
* Code-aware chunking
* Context compression
* Query expansion
* Conversation memory
* More programming-language support
* Streaming responses
* Better repository understanding

## Author

**Raman**

GitHub: `https://github.com/raman-codesX`
