import os
import boto3
import json
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# -------------------------------
# Config
# -------------------------------
OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "rag-index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# AWS Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------------
# OpenSearch Retrieval Model
# -------------------------------
class RetrievalModel:
    def __init__(self, embedder, index_name, opensearch_url):
        self.embedder = embedder
        self.index_name = index_name
        self.opensearch_url = opensearch_url

    def embed_text(self, text):
        return self.embedder.encode(text).tolist()

    def retrieve(self, query, top_k=3):
        query_emb = self.embed_text(query)
        search_query = {
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_emb, "k": top_k}}},
        }
        url = f"{self.opensearch_url}/{self.index_name}/_search"
        res = requests.get(url, json=search_query, auth=("admin", "admin"))
        hits = res.json()["hits"]["hits"]
        return hits

# -------------------------------
# Query Bedrock LLM
# -------------------------------
def query_bedrock(prompt):
    response = bedrock.invoke_model(
        modelId="anthropic.claude-v2",
        body=json.dumps(
            {
                "prompt": prompt,
                "max_tokens_to_sample": 300,
                "temperature": 0.2,
            }
        ),
        contentType="application/json",
    )
    return json.loads(response["body"].read())["completion"]

# -------------------------------
# Streamlit UI
# -------------------------------
retriever = RetrievalModel(embedder, INDEX_NAME, OPENSEARCH_URL)

st.set_page_config(page_title="RAG Query UI", page_icon="💬", layout="centered")
st.title("RAG Query UI")
st.caption("Ask a question and get an answer from RAG (Bedrock + OpenSearch)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
if st.session_state.messages:
    st.write("### History")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

# User input
user_text = st.text_area(
    "Your question", key="user_text", height=120, placeholder="Type something..."
)

col1, col2 = st.columns([1, 1])
with col1:
    send_clicked = st.button(
        "Send", type="primary", use_container_width=True, disabled=not bool(user_text.strip())
    )
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.messages = []
    st.rerun()

# -------------------------------
# Main Flow (Based on your steps)
# -------------------------------
if send_clicked and user_text.strip():
    st.session_state.messages.append({"role": "user", "content": user_text.strip()})

    with st.spinner("Fetching answer from RAG…"):
        try:
            # Step 1: Retrieve docs from OpenSearch
            retrieved_docs = retriever.retrieve(user_text.strip(), top_k=3)

            # Step 2: Convert docs into JSON (id, score, content)
            context_json = json.dumps(
                [
                    {
                        "id": doc["_id"],
                        "score": doc["_score"],
                        "content": doc["_source"]["content"],
                    }
                    for doc in retrieved_docs
                ],
                indent=2,
            )

            # Step 3: Build prompt for Bedrock
            prompt = f"""You are a helpful assistant.
I will give you JSON search results from OpenSearch.
Use them to answer the user’s question clearly and concisely.
If the information is not available, say so.

Search Results (JSON):
{context_json}

Question: {user_text.strip()}
Answer:"""

            # Step 4: Query Bedrock
            answer = query_bedrock(prompt)

            # Step 5: Show answer in chat UI
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {e}"}
            )

    st.rerun()

# -------------------------------
# Optional: Show retrieved docs
# -------------------------------
if send_clicked and user_text.strip():
    with st.expander("🔍 Retrieved Context from OpenSearch"):
        st.json(retrieved_docs)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #808080;
        color: white;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #6e6e6e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

 