import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

# Initialize SentenceTransformer model (same model used for creating embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load FAISS index from disk
def load_faiss_index(index_path: str):
    print(f"Loading FAISS index from {index_path}...")
    faiss_index = FAISS.load_local(index_path)  # Load FAISS index from the saved path
    print("FAISS index loaded successfully!")
    return faiss_index

# Function to generate embedding for a query
def generate_query_embedding(query: str):
    print(f"Generating embedding for the query: '{query}'")
    query_embedding = embedding_model.encode([query])  # Generate query embedding
    return query_embedding

# Function to search the FAISS index for the most similar chunk
def search_faiss_index(query_embedding, faiss_index, top_k=5):
    print(f"Searching FAISS index for the most similar chunks...")
    D, I = faiss_index.index.search(query_embedding, top_k)  # Search for the top k most similar chunks
    return D, I

# Function to display the top k most similar chunks
def display_results(chunks, I, top_k=5):
    print(f"\nTop {top_k} most similar chunks:")
    for idx in I[0]:  # Iterate through the top k results
        print(f"\nChunk {idx + 1}:")
        print(chunks[idx].page_content)  # Print the content of the chunk

def main():
    # Load FAISS index from the saved location
    DB_FAISS_PATH = "embeddings"  # Path where your FAISS index is stored
    faiss_index = load_faiss_index(DB_FAISS_PATH)

    # Sample document chunks (This would typically be loaded from your documents)
    # In the main.py, you should have a list of document chunks from your process.
    # For this example, I am assuming `chunks` is already loaded or passed to the function.
    chunks = []  # Ensure this is populated with your document chunks

    # Input: Query from the user
    user_query = input("Please enter your query: ")  # Get the query from the user
    
    # Step 1: Generate query embedding
    query_embedding = generate_query_embedding(user_query)

    # Step 2: Search the FAISS index for similar chunks
    D, I = search_faiss_index(query_embedding, faiss_index)

    # Step 3: Display the results
    display_results(chunks, I)

if __name__ == "__main__":
    main()
