import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm


def load_chunks(chunk_file):
    chunks = []
    with open(chunk_file, 'r', encoding='utf-8') as f:
        current = []
        for line in f:
            if line.startswith('==== CHUNK'):
                if current:
                    chunks.append(''.join(current).strip())
                    current = []
            else:
                current.append(line)
        if current:
            chunks.append(''.join(current).strip())
    return chunks


if __name__ == "__main__":
    chunk_file = 'chunks/chunks.txt'
    vectordb_file = 'vectordb/faiss.index'
    meta_file = 'vectordb/chunk_metadata.npy'

    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks.")

    embeddings = model.encode(
        chunks, show_progress_bar=True, batch_size=32, normalize_embeddings=True)

    # to save the index with FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if not os.path.exists('vectordb'):
        os.mkdir('vectordb')

    faiss.write_index(index, vectordb_file)
    np.save(meta_file, np.array(chunks))
    print(
        f"Saved FAISS index to {vectordb_file} and chunk meta to {meta_file}")
