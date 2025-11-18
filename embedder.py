import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import numpy as np
import time
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For progress tracking

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load multiple PDFs
def load_multiple_pdfs(folder_path: str):
    docs = []
    print("Loading PDFs from folder:", folder_path)
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())  # Add loaded documents to the list
    print(f"Loaded {len(docs)} documents.")
    return docs

# Split documents into smaller chunks using RecursiveCharacterTextSplitter
def split_documents(docs, chunk_size=1200, chunk_overlap=400):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# Optimized function to generate embeddings for multiple chunks (Batch processing)
def generate_embeddings_for_chunks(chunks, batch_size=32):
    texts = [chunk.page_content for chunk in chunks]  # Extract text content from chunks
    print("Generating embeddings in batches...")

    # Split the text into batches and generate embeddings for each batch
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):  # Using tqdm for progress tracking
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        embeddings.extend(batch_embeddings)  # Append batch embeddings
    
    print(f"Generated embeddings for {len(embeddings)} chunks.")
    return np.array(embeddings)  # Return embeddings as numpy array for FAISS compatibility

# Folder containing your PDFs
document_paths = "/content/pdfs"  # Adjust the folder path as needed

# Step 1: Load PDFs from the folder
docs = load_multiple_pdfs(document_paths)

# Step 2: Split the loaded documents into chunks
chunks = split_documents(docs)

# Step 3: Generate embeddings for all chunks (batch processing)
start_time = time.time()
embeddings = generate_embeddings_for_chunks(chunks)  # Generate embeddings for chunks in batches
texts_and_embeddings = [(chunk.page_content, embedding) for chunk, embedding in zip(chunks, embeddings)]
print(f"Embeddings generated in {time.time() - start_time} seconds.")

# Step 4: Create FAISS index using embeddings
print("Creating FAISS index from the embeddings...")
faiss_index = FAISS.from_embeddings(texts_and_embeddings, embedding_model) # Create FAISS index using embeddings and chunks

# Step 5: Save FAISS index to disk for later use
DB_FAISS_PATH = "embeddings"  # Specify the folder path to save FAISS index
faiss_index.save_local(DB_FAISS_PATH)  # Save FAISS index locally
print("FAISS index saved!")
