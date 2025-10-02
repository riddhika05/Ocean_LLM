from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xarray as xr
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env
load_dotenv()

# --- Initialize FastAPI FIRST (before anything that might crash) ---
app = FastAPI()

# Enable CORS for local development and simple web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NOW Import heavy dependencies ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Assuming VectorStore and embed_text are available from vector_store
from vector_store import VectorStore, embed_text

# --- Initialize Hugging Face LLM for Generation ---
generator = None
tokenizer = None

# Using Flan-T5 Base (approx. 250M parameters)
LLM_MODEL_NAME = "google/flan-t5-base"
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

    # Use CPU
    device = -1
    model = model.to("cpu")
    model.eval()
    print("✓ Model moved to CPU")
    print(f"Device set to use cpu")

    # Use the text2text-generation pipeline
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
DATA_PATH = os.path.join(BASE_DIR, "data", "bay_of_bengal_data.nc")

# Try multiple possible paths
possible_paths = [
    DATA_PATH,
    os.path.join(BASE_DIR, "bay_of_bengal_data.nc"),
    os.path.join(os.path.dirname(BASE_DIR), "data", "bay_of_bengal_data.nc"),
    os.path.join(os.getcwd(), "data", "bay_of_bengal_data.nc"),
    os.path.join(os.getcwd(), "bay_of_bengal_data.nc"),
]

print(f"Searching for ocean data in possible locations...")
ds = None
for path in possible_paths:
    print(f"  Trying: {path}")
    if os.path.exists(path):
        try:
            ds = xr.open_dataset(path)
            DATA_PATH = path
            print(f"✓ Loaded dataset from: {path}")
            print(f"  Variables: {list(ds.data_vars)}")
            print(f"  Dimensions: {dict(ds.dims)}")
            break
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

if ds is None:
    print("\n" + "="*60)
    print("✗ DATASET NOT FOUND!")
    print("="*60)
    print("Please ensure 'bay_of_bengal_data.nc' is in one of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nYou can generate it using the Python script provided earlier.")
    print("="*60 + "\n")

# ----- Create vector store with enhanced data chunks -----
vector_dim = 384  # for 'all-MiniLM-L6-v2'
store = VectorStore(dimension=vector_dim)
texts = []

if ds is not None:
    print("Preparing enhanced ocean data chunks for indexing...")
    
    # Sample spatial points and create descriptive summaries
    # Sample every 10th lat/lon point to reduce data size
    lat_sample = ds.lat.values[::10]
    lon_sample = ds.lon.values[::10]
    
    # Sample times: all time steps (monthly already)
    time_sample = ds.time.values
    
    chunk_count = 0
    for lat in lat_sample[:1]:  # Limit to 10 lat points
        for lon in lon_sample[:1]:  # Limit to 10 lon points
            for time in time_sample[:1]:  # All 12 months
                try:
                    # Extract data at this point
                    point = ds.sel(lat=lat, lon=lon, time=time, method='nearest')
                    
                    # Create descriptive text chunk
                    text = (
                        f"Location: Latitude {float(lat):.2f}°N, Longitude {float(lon):.2f}°E. "
                        f"Date: {pd.Timestamp(time).strftime('%Y-%m-%d')}. "
                    )
                    
                    # Add temperature
                    if 'temperature' in ds:
                        temp = float(point['temperature'].values)
                        text += f"Sea surface temperature: {temp:.2f}°C. "
                    
                    # Add salinity
                    if 'salinity' in ds:
                        sal = float(point['salinity'].values)
                        text += f"Surface salinity: {sal:.2f} PSU. "
                    
                    texts.append(text)
                    chunk_count += 1
                    
                except Exception as e:
                    continue
    
    print(f"Created {len(texts)} descriptive data chunks")
    
    # Index all chunks
    if texts:
        print(f"Indexing {len(texts)} ocean data points...")
        for i, text in enumerate(texts):
            try:
                emb = embed_text(text)
                store.add(text, emb)
                if (i + 1) % 100 == 0:
                    print(f"  Indexed {i + 1}/{len(texts)} chunks...")
            except Exception as e:
                print(f"Error embedding text: {e}")
                break
        print("✓ Vector store ready!")
    else:
        print("✗ No chunks created, vector store remains empty.")
else:
    print("✗ No data loaded, vector store remains empty.")


def extract_query_params(question: str):
    """Extract lat, lon, depth, time parameters from question."""
    params = {}
    
    # Extract latitude
    lat_match = re.search(r'lat(?:itude)?[=\s:]+(-?\d+\.?\d*)', question, re.IGNORECASE)
    if lat_match:
        params['lat'] = float(lat_match.group(1))
    
    # Extract longitude
    lon_match = re.search(r'lon(?:gitude)?[=\s:]+(-?\d+\.?\d*)', question, re.IGNORECASE)
    if lon_match:
        params['lon'] = float(lon_match.group(1))
    
    # Extract depth
    depth_match = re.search(r'depth[=\s:]+(-?\d+\.?\d*)', question, re.IGNORECASE)
    if depth_match:
        params['depth'] = float(depth_match.group(1))
    
    # Extract date (various formats)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
    if date_match:
        params['time'] = date_match.group(1)
    
    # Extract month names
    month_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', question, re.IGNORECASE)
    if month_match:
        month_name = month_match.group(1).capitalize()
        params['month'] = month_name
    
    return params


def query_dataset_directly(question: str, params: dict):
    """Directly query the xarray dataset for specific point queries."""
    if ds is None:
        return None
    
    try:
        # If we have lat/lon, try direct query
        if 'lat' in params and 'lon' in params:
            lat, lon = params['lat'], params['lon']
            
            # Select point
            point = ds.sel(lat=lat, lon=lon, method='nearest')
            
            # Add time if specified
            if 'time' in params:
                point = point.sel(time=params['time'], method='nearest')
            elif 'month' in params:
                # Convert month name to time
                month_map = {
                    'January': '2024-01', 'February': '2024-02', 'March': '2024-03',
                    'April': '2024-04', 'May': '2024-05', 'June': '2024-06',
                    'July': '2024-07', 'August': '2024-08', 'September': '2024-09',
                    'October': '2024-10', 'November': '2024-11', 'December': '2024-12'
                }
                month_time = month_map.get(params['month'])
                if month_time:
                    point = point.sel(time=month_time, method='nearest')
            
            # Build response based on what's being asked
            response = f"At location (lat={lat}°N, lon={lon}°E"
            if 'time' in params:
                response += f", date={params['time']}"
            elif 'month' in params:
                response += f", month={params['month']}"
            response += "):\n"
            
            # Check what variable is being asked about
            if 'temperature' in question.lower() and 'temperature' in point:
                temp = float(point['temperature'].values)
                response += f"Temperature: {temp:.2f}°C"
                return response
            
            elif 'salinity' in question.lower() and 'salinity' in point:
                sal = float(point['salinity'].values)
                response += f"Salinity: {sal:.2f} PSU"
                return response
            
            # If no specific variable, give overview
            if 'temperature' in point:
                temp = float(point['temperature'].values)
                response += f"- Temperature: {temp:.2f}°C\n"
            if 'salinity' in point:
                sal = float(point['salinity'].values)
                response += f"- Salinity: {sal:.2f} PSU\n"
            
            return response.strip()
        
        # Handle aggregate queries
        if 'average' in question.lower() or 'mean' in question.lower():
            if 'temperature' in question.lower():
                avg_temp = float(ds['temperature'].mean().values)
                return f"The average sea surface temperature across all data is {avg_temp:.2f}°C"
            elif 'salinity' in question.lower():
                avg_sal = float(ds['salinity'].mean().values)
                return f"The average surface salinity across all data is {avg_sal:.2f} PSU"
        
        if 'maximum' in question.lower() or 'max' in question.lower() or 'highest' in question.lower():
            if 'temperature' in question.lower():
                max_temp = float(ds['temperature'].max().values)
                max_loc = ds['temperature'].where(
                    ds['temperature'] == ds['temperature'].max(), drop=True
                )
                max_time = pd.Timestamp(max_loc.time.values[0]).strftime('%Y-%m-%d')
                return f"The maximum sea surface temperature is {max_temp:.2f}°C, occurring on {max_time}"
            elif 'salinity' in question.lower():
                max_sal = float(ds['salinity'].max().values)
                return f"The maximum surface salinity is {max_sal:.2f} PSU"
        
        if 'minimum' in question.lower() or 'min' in question.lower() or 'lowest' in question.lower():
            if 'temperature' in question.lower():
                min_temp = float(ds['temperature'].min().values)
                return f"The minimum sea surface temperature is {min_temp:.2f}°C"
            elif 'salinity' in question.lower():
                min_sal = float(ds['salinity'].min().values)
                return f"The minimum surface salinity is {min_sal:.2f} PSU"
        
    except Exception as e:
        print(f"Error in direct query: {e}")
        return None
    
    return None


# ----- API Schemas -----
class QueryRequest(BaseModel):
    question: str


def ask_llm(context: str, question: str) -> str:
    """
    Generate answer using the Flan-T5 model based on retrieved context.
    """
    prompt = (
f"""Based ONLY on the following ocean data points, answer the user's question concisely and factually. 
If the information is not present in the data, state that you cannot answer based on the context.

DATA POINTS:
{context}

QUESTION:
{question}

ANSWER:"""
    )

    try:
        # Check if the generator is the fallback function
        if callable(generator) and not hasattr(generator, "__self__"):
            return generator(prompt)[0]["generated_text"]

        # Flan-T5 (Seq2Seq) generation parameters
        result = generator(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            truncation=True,
        )

        if result and len(result) > 0 and "generated_text" in result[0]:
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
        # Step 1: Try direct dataset query first (for specific point queries)
        params = extract_query_params(request.question)
        direct_answer = query_dataset_directly(request.question, params)
        
        if direct_answer:
            return {
                "answer": direct_answer,
                "method": "direct_query",
                "context_used": []
            }
        
        # Step 2: Fall back to RAG pipeline
        query_emb = embed_text(request.question)
        contexts = store.query(query_emb, top_k=5)
        combined_context = "\n".join(contexts)

        if not combined_context:
            answer = "No relevant context found in the vector store to answer the question."
        else:
            answer = ask_llm(combined_context, request.question)

        return {
            "answer": answer,
            "method": "rag_pipeline",
            "context_used": contexts
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "context_used": []
        }


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Bay of Bengal Ocean RAG API is running",
        "endpoints": {
            "POST /query": "Submit a question about ocean data",
            "GET /health": "Check API health",
            "GET /dataset/info": "Get dataset information"
        }
    }


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
        "data_points": len(texts),
        "dataset_loaded": ds is not None,
        "dataset_variables": list(ds.data_vars) if ds is not None else [],
        "dataset_path": DATA_PATH if ds is not None else "not found"
    }


@app.get("/dataset/info")
def dataset_info():
    """Get information about the loaded dataset."""
    if ds is None:
        return {"error": "Dataset not loaded"}
    
    return {
        "variables": list(ds.data_vars),
        "dimensions": dict(ds.dims),
        "coords": {k: {"min": float(v.min()), "max": float(v.max())} 
                   for k, v in ds.coords.items()},
        "attributes": dict(ds.attrs)
    }