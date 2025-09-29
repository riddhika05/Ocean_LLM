import faiss
import numpy as np
from openai import OpenAI

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)  # FAISS index (L2 distance)
        self.texts = []  # Keep original texts for retrieval

    def add(self, text, embedding):
        self.index.add(np.array([embedding], dtype='float32'))  # Store vector
        self.texts.append(text)  # Save original text

    def query(self, embedding, top_k=1):
        D, I = self.index.search(np.array([embedding], dtype='float32'), top_k)
        return [self.texts[i] for i in I[0]]  # Return closest text(s)

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Low-cost embedding model
        input=text
    )
    return response['data'][0]['embedding']

