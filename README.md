# 📄 RAG Chatbot: Document QA with Streaming LLM Responses

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** over a large legal document (eBay User Agreement), enabling users to ask natural language questions and receive fact-grounded, real-time answers.  
The pipeline leverages semantic chunking, fast vector search (FAISS), and a locally-running instruction-tuned LLM (Mistral-7B via Ollama) for efficient and accurate document QA.

---

## 📐 Project Architecture & Flow

```mermaid
graph TD
    A[Document Upload] --> B[Preprocessing & Cleaning]
    B --> C[Sentence-aware Chunking]
    C --> D[Embedding (MiniLM)]
    D --> E[Vector DB Storage (FAISS)]
    E --> F[Chat Interface (Streamlit)]
    F --> G[Query --> Embed --> Retrieve Top Chunks]
    G --> H[Prompt LLM (Mistral/Ollama) with Context]
    H --> I[Streamed, Grounded Answer]

    Pipeline Steps:
Preprocessing: Clean text, remove formatting, split into overlapping, sentence-aligned chunks.
Embedding: Generate semantic vector for each chunk using all-MiniLM-L6-v2.
Vector DB: Store chunk vectors in FAISS for fast similarity search.
Retrieval + Generation: At query time, select top chunks, inject into prompt, and generate a streaming response with local LLM via Ollama.
User Interface: Streamlit app with live response, chunk sourcing, sidebar info, and full chat history.

🛠️ Setup & Running the Project Locally
1. Install Python dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt

2. Install Ollama (for fast, local LLM inference)
Download from ollama.com/download and install for your OS.

3. Download the Mistral model
ollama pull mistral

4. Place your document in the data/ folder
E.g., data/AI Training Document.txt

5. Preprocess and embed the document
python src/chunk_document.py
python src/embed_and_index.py

Or, run:
jupyter notebook notebooks/01_preprocessing_and_embedding.ipynb
(for step-by-step, interactive data exploration and building)

6. Start Ollama for LLM serving
ollama run mistral

(Leave this running in a terminal.)
7. Start the Streamlit Chatbot
streamlit run app.py
Visit http://localhost:8501 in your browser.


🤖 Model & Embedding Choices
Embedding Model:
all-MiniLM-L6-v2 (Sentence Transformers)
Fast, accurate representation for semantic search.
Vector Database:
FAISS (IndexFlatIP; cosine similarity; CPU-based for small-to-mid scale).
LLM:
Mistral-7B-Instruct (run locally via Ollama)
Provides high-quality, instruction-following answers.
Efficient GPU/CPU use, real-time token streaming.

Prompt Template Used:

This template forces the LLM to ground all answers in retrieved text only—minimizing hallucinations.
🗂️ Project Structure

🏃 End-to-End Pipeline Steps
Document preprocessing:
Clean & chunk with src/chunk_document.py or the Jupyter notebook.
Embeddings and DB creation:
Use src/embed_and_index.py to create MiniLM embeddings and FAISS vector store.
Launch chatbot:
Ensure Ollama is running Mistral; start Streamlit via streamlit run app.py.

🧑‍💻 Sample Queries & Screenshots
Example Queries
Query	AI Response (Excerpt)

How are legal disputes resolved between eBay and users?	



What are sellers responsible for when listing items?	



Who is eBay’s CEO? (failure)	


Screenshots
data/Screenshot 2025-06-24 at 17.58.47.png
Chatbot Demo Screenshot
Chatbot showing real-time streaming answer and cited chunks.
Link to demo video (replace with actual URL if submitting)


🚦 Notes: Hallucination, Limitations, Performance
Control:
Strong prompt instructions + chunk-based retrieval keep answers accurate and grounded.
Limitations:
Info must be present in document; ambiguous queries may retrieve wrong passages if chunking misses key context.
Reproducibility:
Full pipeline in scripts and Jupyter notebook; step-by-step documentation for review.

📖 Further Details
The notebook in /notebooks presents preprocessing, chunking, and embedding steps in detail.
All source code is documented and modular for easy extension.