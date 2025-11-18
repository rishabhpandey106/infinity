# PDF Retrieval and Question Answering System

This project builds a simple and practical pipeline for reading PDF documents, turning them into searchable embeddings, and answering user questions through a Streamlit interface. Everything is kept straightforward: PDFs become text, text becomes chunks, chunks become vectors, vectors go into FAISS, and questions pull out the most relevant pieces.

In this we have identified the need of using RAG for our Institute Policies like Plag Policy, Green Policy, etc. The reason is Policies of an organization can change and model may hallucinate and does not provide accurate answer.

--------------------------------------------------------------------

# Folder Structure

Case2_RagPoweredApp/  
│  
├─ chains/  
│   ├─ qa_chain.py  
│   └─ utils.py  
│  
├─ embeddings/  
│   └─ vector_store.py  
│  
├─ loaders/  
│   └─ pdf_loader.py  
│  
├─ PDF_document/  
│   └─ data1.pdf
│   └─ data2.pdf
│   └─ data3.pdf
│   └─ data4.pdf
│   
│  
├─ app.py  
├─ faiss_embed.py  
├─ main.py  
│  
├─ requirements.txt  
├─ .env  
└─ README.md  

--------------------------------------------------------------------

# Architecture
1. **PDF Loading**  
   The PDFs inside the PDF_document folder are read using a loader.  
   The loader extracts raw text and hands it over to the next stage.
   This is done using PyPDFLoader to load python and then created a doc file. 

3. **Chunk Creation**  
   The extracted text is split into smaller, meaningful chunks.  
   This avoids oversized inputs and keeps context clean.

4. **Embedding**  
   Each chunk is passed through an embedding model.  
   The output is a vector that represents the meaning of the chunk.

6. **FAISS Vector Store**  
   All vectors are collected and stored inside a FAISS.index.  
   This creates a searchable database where similar meanings sit closer together.

7. **Query Embedding**  
   When the user types a question in the Streamlit interface,  
   it is also turned into a vector using same embedding model.

8. **Semantic Retrieval**  
   FAISS compares the query vector with all stored vectors  
   and returns the closest matches—the k most relevant chunks using similarity.

9. **Answer Formation**  
   The retrieved chunks and the user’s question are combined,  
   producing a response that stays grounded in the PDF content.


--------------------------------------------------------------------

# Setup Instructions

## Install Dependencies

pip install -r requirements.txt (better to run in virtual env.)


## Build the Vector Index

This step reads your PDFs, creates chunks, embeds them, and stores them in FAISS.
python faiss_embed.py


## Start the Streamlit Application

This launches the user interface where you can ask questions.
streamlit run app.py

## Deploye on Vercel/Render/Streamlit


--------------------------------------------------------------------

# Summary

Once everything is set up, the pipeline becomes very direct:  
PDFs go in → embeddings are built → FAISS stores them → Streamlit lets you ask questions → the system returns answers backed by your documents.  
Everything runs locally, stays simple and works end-to-end without extra complexity.


