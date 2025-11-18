import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st  # Import Streamlit
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize SentenceTransformer model (same model used for creating embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load FAISS index from disk
def load_faiss_index(index_path: str):
    faiss_index = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return faiss_index

# Function to generate embedding for a query
def generate_query_embedding(query: str):
    query_embedding = embedding_model.encode([query])  # Generate query embedding
    return query_embedding

# Function to search the FAISS index for the most similar chunk
def search_faiss_index(query_embedding, faiss_index, top_k=3):
    D, I = faiss_index.index.search(query_embedding, top_k)  # Perform the search
    return D, I

# Streamlit UI
def main():
    st.title("Document Search with FAISS")
    
    DB_FAISS_PATH = "faiss"  # Path where your FAISS index is saved
    faiss_index = load_faiss_index(DB_FAISS_PATH)

    chunks = []  

    query = st.text_input("Please enter your query:") 

    if query:
        # Prefer using the FAISS vectorstore API which returns stored Document objects
        try:
            results = faiss_index.similarity_search(query, k=3)
        except Exception:
            # Fallback: perform raw index search and map indices to the docstore
            query_embedding = generate_query_embedding(query)
            D, I = search_faiss_index(query_embedding, faiss_index)

            results = []
            index_map = getattr(faiss_index, "index_to_docstore_id", None)
            docstore = getattr(faiss_index, "docstore", None)
            if index_map and docstore:
                for idx in I[0]:
                    if idx < len(index_map):
                        doc_id = index_map[idx]
                        # InMemoryDocstore usually keeps docs in a private dict
                        if hasattr(docstore, "_dict"):
                            doc = docstore._dict.get(doc_id)
                        else:
                            # Generic mapping fallback
                            try:
                                doc = docstore.get(doc_id)
                            except Exception:
                                doc = None
                        if doc:
                            results.append(doc)

        if results:
            st.write(f"Top {len(results)} most similar chunks:")
            for i, doc in enumerate(results):
                st.write(f"### Chunk {i + 1}")
                st.write(doc.page_content)
        else:
            st.write("No valid results found. Please try with a different query.")


if __name__ == "__main__":
    main()
