import faiss
import numpy as np
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# --- Initialize Gemini Client for Embeddings ---
print("Initializing Gemini Client for embeddings...")
try:
    client = genai.Client()
    EMBEDDING_MODEL = "text-embedding-004"
    print(f"✓ Gemini embedding client initialized with model: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"✗ Failed to initialize Gemini Client: {e}")
    client = None
    EMBEDDING_MODEL = None


class VectorStore:
    def __init__(self, dimension):
        # The dimension must match the output size of the chosen embedding model
        # text-embedding-004 produces 768-dimensional vectors
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
        
        if self.index.ntotal == 0:
            print("Warning: Vector store is empty!")
            return []
            
        D, I = self.index.search(embedding.reshape(1, -1).astype('float32'), top_k)
        return [self.texts[i] for i in I[0]]  # Return closest text(s)


# --- Embedding Function using Gemini API ---
def embed_text(text):
    """
    Generate embeddings using Gemini's text-embedding-004 model.
    Returns a 768-dimensional numpy array.
    """
    if client is None:
        raise RuntimeError("Gemini client not initialized. Cannot generate embeddings.")
    
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )
        
        # Extract embedding values
        embedding = response.embeddings[0].values
        
        # Convert to numpy array
        embedding_array = np.array(embedding, dtype='float32')
        
        return embedding_array
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

# Note: The 'text-embedding-004' model produces a 768-dimensional vector.
# You MUST update the vector_dim in main.py to 768.