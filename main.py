import os
import faiss
import base64
import requests
import streamlit as st

from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# PAGE CONFIG

st.set_page_config(
    page_title="CodeBase AI",
    layout="wide"
)

# SESSION STATE


if "messages" not in st.session_state:
    st.session_state.messages = []


# TOP HEADER

header_left, header_right = st.columns([6, 1])

with header_left:
    st.title("◈ CodeBase AI")
    st.caption("AI-powered GitHub codebase analysis")

with header_right:
    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.divider()


# HERO SECTION


st.markdown("### Understand your codebase")

st.write(
    "Connect a GitHub repository and ask questions about "
    "your code using AI-powered retrieval."
)

st.write("")

# LOAD ENVIRONMENT VARIABLES

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")


# =========================================================
# REPOSITORY SECTION
# =========================================================

st.subheader("🔗 Repository")

file_url = st.text_input(
    "GitHub Repository URL",
    value="https://github.com/raman-codesX/mini-project.git",
    placeholder="https://github.com/username/repository"
)


# =========================================================
# REPOSITORY INFO
# =========================================================

repo_col1, repo_col2, repo_col3 = st.columns(3)

with repo_col1:
    st.metric("Source", "GitHub")

with repo_col2:
    st.metric("Search", "Hybrid")

with repo_col3:
    st.metric("AI", "Gemini")


st.divider()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return model, reranker


model, reranker = load_models()


# =========================================================
# PREVIOUS CHAT
# =========================================================

if st.session_state.messages:

    st.subheader("Conversation")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("📄 View Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"• {source}"
                        )

    st.divider()


# =========================================================
# QUESTION SECTION
# =========================================================

st.subheader("Ask your codebase")

query = st.text_area(
    "Your question",
    placeholder=(
        "Example: How does the RAG pipeline work?"
    ),
    height=110
)


ask_col1, ask_col2, ask_col3 = st.columns(
    [4, 1, 4]
)

with ask_col2:

    ask_button = st.button(
        "Ask →",
        use_container_width=True
    )


# =========================================================
# RUN RAG
# =========================================================

if ask_button and query.strip():

    query = query.strip()


    # =====================================================
    # SAVE USER QUESTION
    # =====================================================

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    # =====================================================
    # LOADING
    # =====================================================

    with st.spinner(
        "Analyzing repository and finding relevant code..."
    ):

        # -------------------------------------------------
        # 1. PARSE GITHUB URL
        # -------------------------------------------------

        parts = file_url.rstrip("/").split("/")

        owner = parts[-2]

        repo = parts[-1].removesuffix(".git")


        # -------------------------------------------------
        # 2. GITHUB HEADERS
        # -------------------------------------------------

        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json"
        }


        # -------------------------------------------------
        # 3. GET REPOSITORY INFORMATION
        # -------------------------------------------------

        repo_api = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}"
        )

        response = requests.get(
            repo_api,
            headers=headers
        )

        repo_data = response.json()


        if response.status_code != 200:

            st.error("GitHub API Error!")

            st.json(repo_data)

            st.stop()


        branch = repo_data["default_branch"]


        # -------------------------------------------------
        # 4. GET ALL FILES
        # -------------------------------------------------

        api_url = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}/git/trees/"
            f"{branch}?recursive=1"
        )

        response = requests.get(
            api_url,
            headers=headers
        )

        data = response.json()


        if response.status_code != 200:

            st.error("Tree API Error!")

            st.json(data)

            st.stop()


        # -------------------------------------------------
        # 5. READ FILES
        # -------------------------------------------------

        documents = []

        for item in data["tree"]:

            if item["type"] == "blob":

                path = item["path"]


                if path.endswith(
                    (".py", ".md", ".txt")
                ):

                    File_url = (
                        f"https://api.github.com/repos/"
                        f"{owner}/{repo}/contents/"
                        f"{path}"
                    )


                    response = requests.get(
                        File_url,
                        headers=headers
                    )

                    file_data = response.json()


                    if response.status_code != 200:
                        continue


                    if "content" not in file_data:
                        continue


                    content = base64.b64decode(
                        file_data["content"]
                    ).decode("utf-8")


                    documents.append({
                        "path": path,
                        "content": content
                    })


        # -------------------------------------------------
        # 6. CHUNKING + METADATA
        # -------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = []

        metadata = []


        for document in documents:

            file_chunk = splitter.split_text(
                document["content"]
            )


            for chunk in file_chunk:

                chunks.append(chunk)

                metadata.append({
                    "source": document["path"]
                })


        # -------------------------------------------------
        # 7. CREATE EMBEDDINGS
        # -------------------------------------------------

        embedding = model.encode(
            chunks,
            convert_to_numpy=True
        )


        # -------------------------------------------------
        # 8. FAISS
        # -------------------------------------------------

        dimension = embedding.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(embedding)


        # -------------------------------------------------
        # 9. BM25
        # -------------------------------------------------

        tokenized_chunks = [
            chunk.lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_chunks
        )


        # -------------------------------------------------
        # 10. VECTOR SEARCH
        # -------------------------------------------------

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )


        vector_distance, vector_indices = index.search(
            query_embedding,
            5
        )


        vector_result = []


        for i in vector_indices[0]:

            vector_result.append({

                "text": chunks[i],

                "source": metadata[i]["source"],

                "type": "vector"

            })


        # -------------------------------------------------
        # 11. BM25 SEARCH
        # -------------------------------------------------

        bm25_score = bm25.get_scores(
            query.lower().split()
        )


        bm25_indices = sorted(
            range(len(bm25_score)),
            key=lambda i: bm25_score[i],
            reverse=True
        )[:5]


        bm25_result = []


        for i in bm25_indices:

            bm25_result.append({

                "text": chunks[i],

                "source": metadata[i]["source"],

                "type": "bm25"

            })


        # -------------------------------------------------
        # 12. HYBRID SEARCH
        # -------------------------------------------------

        hybrid_result = []

        seen = set()


        for result in (
            vector_result + bm25_result
        ):

            text = result["text"]


            if text not in seen:

                seen.add(text)

                hybrid_result.append(
                    result
                )


        # -------------------------------------------------
        # 13. RE-RANKER
        # -------------------------------------------------

        pair = [

            [query, result["text"]]

            for result in hybrid_result

        ]


        scores = reranker.predict(
            pair
        )


        for result, score in zip(
            hybrid_result,
            scores
        ):

            result["reranker_score"] = float(
                score
            )


        hybrid_result.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )


        top_result = hybrid_result[:3]


        # -------------------------------------------------
        # 14. BUILD CONTEXT
        # -------------------------------------------------

        context_parts = []


        for result in top_result:

            context_parts.append(
                f"""
SOURCE: {result["source"]}

{result["text"]}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        # -------------------------------------------------
        # 15. GEMINI
        # -------------------------------------------------

        client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            )
        )


        response = client.chat.completions.create(

            model="gemini-3.6-flash",

            messages=[

                {
                    "role": "system",

                    "content": """
You are a codebase assistant.

Answer the user's question using only
the provided repository context.

If the context is insufficient,
say that the repository does not provide
enough information.

Always mention the relevant source file
when possible.
"""
                },

                {
                    "role": "user",

                    "content": f"""
REPOSITORY CONTEXT:

{context}


QUESTION:

{query}
"""
                }

            ]

        )


        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )


        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        sources = [

            result["source"]

            for result in top_result

        ]


        # -------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # -------------------------------------------------

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer,

            "sources": sources

        })


    # =====================================================
    # REFRESH UI
    # =====================================================

    st.rerun()


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

if not st.session_state.messages:

    st.divider()

    st.subheader("Try asking")

    example_col1, example_col2, example_col3 = st.columns(3)


    with example_col1:

        st.info(
            "🔍 **Repository Overview**\n\n"
            "What does this project do?"
        )


    with example_col2:

        st.info(
            "🧠 **Code Analysis**\n\n"
            "How does the RAG pipeline work?"
        )


    with example_col3:

        st.info(
            "🐛 **Debugging**\n\n"
            "Where is the GitHub API used?"
        )


# =========================================================
# DEBUG
# =========================================================

with st.expander("⚙️Token Information"):

    st.write(
        "GitHub token exists:",
        github_token is not None
    )

    st.write(
        "GitHub token length:",
        len(github_token)
        if github_token
        else 0
    )

