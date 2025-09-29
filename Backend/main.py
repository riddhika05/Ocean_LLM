from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xarray as xr
import pandas as pd
import os
from dotenv import load_dotenv

# --- Hugging Face Imports ---
# Using AutoModelForSeq2SeqLM for T5 (Encoder-Decoder) architecture
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
# ----------------------------

# Assuming VectorStore and embed_text are available from vector_store
from vector_store import VectorStore, embed_text

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Enable CORS for local development and simple web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Hugging Face LLM for Generation ---
generator = None
tokenizer = None

# Using Flan-T5 Small (approx. 80M parameters) - highly lightweight and effective for RAG
LLM_MODEL_NAME = "google/flan-t5-small"
print(f"Loading model: {LLM_MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    print("✓ Tokenizer loaded")

    # Use AutoModelForSeq2SeqLM for Flan-T5
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float32,
    )
    print("✓ Model weights loaded")

    # Use CPU (-1) since the model is small and this is safer for general environments
    device = -1
    model = model.to("cpu")
    model.eval()
    print("✓ Model moved to CPU")

    # Use the text2text-generation pipeline for sequence-to-sequence models
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"✓ Pipeline created - Model {LLM_MODEL_NAME} loaded successfully!")

except Exception as e:
    import traceback

    print(f"✗ Failed to load model: {e}")
    print(traceback.format_exc())

    # Fallback minimal generator
    def _fallback_generator(prompt, **kwargs):
        return [
            {
                "generated_text": "Error: Model loading failed. Cannot generate response. Please check logs."
            }
        ]

    generator = _fallback_generator

# ----- Read NetCDF & flatten to text -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ocean_data.nc")

if not os.path.exists(DATA_PATH):
    # Adjust path for common execution setups
    DATA_PATH = os.path.join(BASE_DIR, "..", "Backend", "data", "ocean_data.nc")
    DATA_PATH = os.path.abspath(DATA_PATH)

print(f"Loading ocean data from: {DATA_PATH}")

try:
    ds = xr.open_dataset(DATA_PATH)
    df = ds.to_dataframe().reset_index()

    texts = []
    for _, row in df.iterrows():
        texts.append(
            f"Lat={row['lat']}, Lon={row['lon']}, "
            f"Temp={row.get('temperature', 'NA')}C, "
            f"Salinity={row.get('salinity', 'NA')}ppt"
        )
except Exception as e:
    print(f"✗ Failed to load or process data: {e}")
    df = pd.DataFrame()
    texts = []


# ----- Create vector store (assuming VectorStore and embed_text are defined elsewhere) -----
vector_dim = 384  # for 'all-MiniLM-L6-v2' (the typical embedding model)
store = VectorStore(dimension=vector_dim)

if texts:
    print(f"Indexing {min(len(texts), 100)} ocean data points...")
    # Indexing only the first 100 points for speed
    for text in texts[:100]:
        try:
            emb = embed_text(text)
            store.add(text, emb)
        except Exception as e:
            print(f"Error embedding text: {e}")
            break
    print("✓ Vector store ready!")
else:
    print("✗ No data loaded, vector store remains empty.")

# ----- API Schemas -----
class QueryRequest(BaseModel):
    question: str


def ask_llm(context: str, question: str) -> str:
    """
    Generate answer using the Flan-T5 model based on retrieved context.

    Improved, directive prompt structure for better RAG performance.
    """
    # ----------------------------------------------------------------------
    # BETTER PROMPT: Clear instructions, constraints, and delimiters
    # ----------------------------------------------------------------------
    prompt = (
f"""Based ONLY on the following ocean data points, answer the user's question. 
If the information is not present in the data, state that you cannot answer based on the context.
Keep your answer extremely concise and factual.

DATA POINTS:
{context}

QUESTION:
{question}

ANSWER:"""
    )
    # ----------------------------------------------------------------------

    try:
        # Check if the generator is the fallback function
        if callable(generator) and not hasattr(generator, "__self__"):
            return generator(prompt)[0]["generated_text"]

        # Flan-T5 (seq2seq) generation parameters
        result = generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            # For seq2seq models like T5, return_full_text is generally irrelevant
            # as it returns the output sequence, not the full input/output combined.
            truncation=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        if result and len(result) > 0 and "generated_text" in result[0]:
            # For Flan-T5, the result should be the direct, concise answer
            answer = result[0]["generated_text"].strip()
            return answer if answer else "No answer could be generated by the model."

        return "Could not generate an answer based on the provided data."

    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error during LLM generation: {str(e)}"


@app.post("/query")
def query(request: QueryRequest):
    """Handle query requests with RAG pipeline."""
    try:
        query_emb = embed_text(request.question)
        contexts = store.query(query_emb, top_k=3)
        combined_context = "\n".join(contexts)

        if not combined_context:
            answer = "No relevant context found in the vector store to answer the question."
        else:
            answer = ask_llm(combined_context, request.question)

        return {"answer": answer, "context_used": contexts}
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "context_used": []}


@app.get("/")
def root():
    return {"status": "ok", "message": "Ocean RAG API is running"}


@app.get("/health")
def health():
    if generator and hasattr(generator, "__class__"):
        model_status = (
            "loaded" if "Pipeline" in generator.__class__.__name__ else "fallback"
        )
    else:
        model_status = "fallback"

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_name": LLM_MODEL_NAME if model_status == "loaded" else "none",
        "data_points": len(texts) if 'texts' in locals() else 0,
    }
