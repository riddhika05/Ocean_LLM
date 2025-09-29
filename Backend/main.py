from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xarray as xr
import pandas as pd
from vector_store import VectorStore, embed_text
import os
from dotenv import load_dotenv

# --- Hugging Face Imports ---
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# ---------------------------

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Enable CORS for local development and simple web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Hugging Face LLM for Generation (with safe fallback) ---
generator = None
tokenizer = None

try:
    # Using distilgpt2 - smaller and faster for CPU
    LLM_MODEL_NAME = "distilgpt2"
    print(f"Loading model: {LLM_MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    print("✓ Tokenizer loaded")
    
    # Set pad_token to eos_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float32,
    )
    print("✓ Model weights loaded")
    
    # Move model to CPU explicitly
    model = model.to('cpu')
    model.eval()  # Set to evaluation mode
    print("✓ Model moved to CPU")
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # -1 means CPU
    )
    print(f"✓ Pipeline created - Model {LLM_MODEL_NAME} loaded successfully!")
    
except Exception as e:
    import traceback
    print(f"✗ Failed to load model: {e}")
    print(traceback.format_exc())
    
    # Fallback minimal generator
    def _fallback_generator(prompt, **kwargs):
        return [{"generated_text": "I cannot load the large model locally. Based on provided data, here is a brief summary."}]
    
    generator = _fallback_generator


# ----- Read NetCDF & flatten to text -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ocean_data.nc")

# For repo layout where data is Backend/data/ocean_data.nc
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "..", "Backend", "data", "ocean_data.nc")
    DATA_PATH = os.path.abspath(DATA_PATH)

print(f"Loading ocean data from: {DATA_PATH}")
ds = xr.open_dataset(DATA_PATH)
df = ds.to_dataframe().reset_index()

texts = []
for _, row in df.iterrows():
    texts.append(
        f"Lat={row['lat']}, Lon={row['lon']}, "
        f"Temp={row.get('temperature', 'NA')}C, "
        f"Salinity={row.get('salinity', 'NA')}ppt"
    )

# ----- Create vector store -----
vector_dim = 384  # for 'all-MiniLM-L6-v2'
store = VectorStore(dimension=vector_dim)

print(f"Indexing {min(len(texts), 100)} ocean data points...")
for text in texts[:100]:
    emb = embed_text(text)
    store.add(text, emb)
print("✓ Vector store ready!")


# ----- API -----
class QueryRequest(BaseModel):
    question: str


def ask_llm(context, question):
    """Generate answer using LLM based on retrieved context."""
    # Simplified prompt for better generation
    prompt = f"Ocean Data: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        # Check if we're using the real generator or fallback
        if callable(generator) and not hasattr(generator, '__self__'):
            # It's the fallback function
            result = generator(prompt)
        else:
            # It's the real pipeline - use return_full_text=True
            result = generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                return_full_text=True,  # Changed to True
                truncation=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        if result and len(result) > 0 and 'generated_text' in result[0]:
            full_text = result[0]['generated_text'].strip()
            
            # Extract only the generated part (after the prompt)
            if "Answer:" in full_text:
                # Get everything after "Answer:"
                parts = full_text.split("Answer:", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Take first sentence or up to newline
                    answer = answer.split('\n')[0].strip()
                    return answer if answer else "No clear answer found in data."
            
            return "Could not extract answer from generation."
        
        return "Could not generate an answer based on the provided data."
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: {str(e)}"


@app.post("/query")
def query(request: QueryRequest):
    """Handle query requests with RAG pipeline."""
    try:
        # Get embedding for the question
        query_emb = embed_text(request.question)
        
        # Retrieve top 3 most relevant contexts
        contexts = store.query(query_emb, top_k=3)
        combined_context = "\n".join(contexts)
        
        # Generate answer using LLM
        answer = ask_llm(combined_context, request.question)
        
        return {
            "answer": answer,
            "context_used": contexts
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "context_used": []
        }


@app.get("/")
def root():
    return {"status": "ok", "message": "Ocean RAG API is running"}


@app.get("/health")
def health():
    # Check if generator is a pipeline (has __class__.__name__)
    if generator and hasattr(generator, '__class__'):
        model_status = "loaded" if 'Pipeline' in generator.__class__.__name__ else "fallback"
    else:
        model_status = "fallback"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "model_name": "distilgpt2" if model_status == "loaded" else "none",
        "data_points": len(texts)
    }