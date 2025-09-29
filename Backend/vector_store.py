import faiss
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

# --- Initialize Hugging Face Embedding Model ---
# Using a highly-rated, fast embedding model as an example.
# This model will be downloaded the first time it's run.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


class VectorStore:
    def __init__(self, dimension):
        # The dimension must match the output size of the chosen embedding model
        self.index = faiss.IndexFlatL2(dimension)  # FAISS index (L2 distance)
        self.texts = []  # Keep original texts for retrieval

    def add(self, text, embedding):
        # Ensure embedding is a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype='float32')
        self.index.add(embedding.reshape(1, -1).astype('float32'))  # Store vector
        self.texts.append(text)  # Save original text

    def query(self, embedding, top_k=1):
        # Ensure embedding is a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype='float32')
        D, I = self.index.search(embedding.reshape(1, -1).astype('float32'), top_k)
        return [self.texts[i] for i in I[0]]  # Return closest text(s)


# --- Embedding Function using Hugging Face Model ---
def embed_text(text):
    # The SentenceTransformer model returns a numpy array
    embedding = embedding_model.encode(text)
    return embedding  # Return as numpy array for consistency

# Note: The 'all-MiniLM-L6-v2' model produces a 384-dimensional vector.
# You MUST update the vector_dim in main.py to 384.