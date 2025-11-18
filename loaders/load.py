from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# function to load pdf from folder path
def load_multiple_pdfs(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        
        if file.endswith(".pdf"):
            
            loader = PyPDFLoader(os.path.join(folder_path, file))
            
            docs.extend(loader.load())
    
    print(f"Loaded {len(docs)} documents.")
    return docs

# Split docs into chunks uing RecursiveCharacterTextSplitter
def split_documents(docs, chunk_size=1200, chunk_overlap=400):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

document_paths = "C:\\Users\\Rishabh\\Desktop\\hack\\rag\\pdfs"

docs = load_multiple_pdfs(document_paths)
chunks = split_documents(docs)