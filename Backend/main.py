from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xarray as xr
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import re

# --- Hugging Face Imports ---
# Using AutoModelForSeq2SeqLM for Flan-T5 (Encoder-Decoder) architecture
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

# Using Flan-T5 Base (approx. 250M parameters) - the best balance of performance and size for RAG
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
DATA_PATH = os.path.join(BASE_DIR, "data", "india_ocean_data.nc")

# Try multiple possible paths
possible_paths = [
    DATA_PATH,
    os.path.join(BASE_DIR, "india_ocean_data.nc"),
    os.path.join(os.path.dirname(BASE_DIR), "data", "india_ocean_data.nc"),
    os.path.join(os.getcwd(), "data", "indian_ocean_data.nc"),
    os.path.join(os.getcwd(), "india_ocean_data.nc"),
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
    print("Please ensure 'india_ocean_data.nc' is in one of these locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nYou can generate it using the Python script provided earlier.")
    print("="*60 + "\n")

# ----- Create vector store with enhanced data chunks -----
vector_dim = 384  # for 'all-MiniLM-L6-v2' (the typical embedding model)
store = VectorStore(dimension=vector_dim)
texts = []

if ds is not None:
    print("Preparing enhanced ocean data chunks for indexing...")
    
    # Sample spatial points and create descriptive summaries
    # Sample every 10th lat/lon point to reduce data size
    lat_sample = ds.lat.values[::10]
    lon_sample = ds.lon.values[::10]
    
    # Sample times: monthly (every 30 days)
    time_sample = ds.time.values[::30]
    
    chunk_count = 0
    for lat in lat_sample[:10]:  # Limit to 10 lat points
        for lon in lon_sample[:10]:  # Limit to 10 lon points
            for time in time_sample[:12]:  # 12 months
                try:
                    # Extract data at this point
                    point = ds.sel(lat=lat, lon=lon, time=time, method='nearest')
                    
                    # Create descriptive text chunk
                    text = (
                        f"Location: Latitude {float(lat):.2f}°N, Longitude {float(lon):.2f}°E. "
                        f"Date: {pd.Timestamp(time).strftime('%Y-%m-%d')}. "
                    )
                    
                    # Add surface data
                    if 'sea_surface_temperature' in ds:
                        sst = float(point['sea_surface_temperature'].values)
                        text += f"Sea surface temperature: {sst:.2f}°C. "
                    
                    if 'temperature' in ds:
                        surf_temp = float(point['temperature'].sel(depth=0, method='nearest').values)
                        text += f"Surface temperature: {surf_temp:.2f}°C. "
                        
                        # Add temperature at depth
                        if 100 in ds.depth.values or len(ds.depth) > 3:
                            deep_temp = float(point['temperature'].sel(depth=100, method='nearest').values)
                            text += f"Temperature at 100m depth: {deep_temp:.2f}°C. "
                    
                    if 'salinity' in ds:
                        surf_sal = float(point['salinity'].sel(depth=0, method='nearest').values)
                        text += f"Surface salinity: {surf_sal:.2f} PSU. "
                    
                    if 'chlorophyll' in ds:
                        chl = float(point['chlorophyll'].values)
                        text += f"Chlorophyll-a: {chl:.3f} mg/m³. "
                    
                    if 'u_current' in ds and 'v_current' in ds:
                        u = float(point['u_current'].sel(depth=0, method='nearest').values)
                        v = float(point['v_current'].sel(depth=0, method='nearest').values)
                        speed = np.sqrt(u**2 + v**2)
                        text += f"Surface current speed: {speed:.2f} m/s. "
                    
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
    lat_match = re.search(r'lat(?:itude)?[=\s]+(-?\d+\.?\d*)', question, re.IGNORECASE)
    if lat_match:
        params['lat'] = float(lat_match.group(1))
    
    # Extract longitude
    lon_match = re.search(r'lon(?:gitude)?[=\s]+(-?\d+\.?\d*)', question, re.IGNORECASE)
    if lon_match:
        params['lon'] = float(lon_match.group(1))
    
    # Extract depth
    depth_match = re.search(r'depth[=\s]+(-?\d+\.?\d*)', question, re.IGNORECASE)
    if depth_match:
        params['depth'] = float(depth_match.group(1))
    
    # Extract date
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
    if date_match:
        params['time'] = date_match.group(1)
    
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
            
            # Add depth if specified
            if 'depth' in params and 'temperature' in ds:
                depth = params['depth']
                point = point.sel(depth=depth, method='nearest')
            
            # Build response based on what's being asked
            response = f"At location (lat={lat}, lon={lon}"
            if 'depth' in params:
                response += f", depth={params['depth']}m"
            if 'time' in params:
                response += f", date={params['time']}"
            response += "):\n"
            
            # Check what variable is being asked about
            if 'temperature' in question.lower() and 'temperature' in point:
                temp = float(point['temperature'].values) if 'depth' in params else float(point['temperature'].sel(depth=0).values)
                response += f"Temperature: {temp:.2f}°C"
                return response
            
            elif 'salinity' in question.lower() and 'salinity' in point:
                sal = float(point['salinity'].values) if 'depth' in params else float(point['salinity'].sel(depth=0).values)
                response += f"Salinity: {sal:.2f} PSU"
                return response
            
            elif 'chlorophyll' in question.lower() and 'chlorophyll' in point:
                chl = float(point['chlorophyll'].values)
                response += f"Chlorophyll-a: {chl:.3f} mg/m³"
                return response
            
            elif 'current' in question.lower():
                if 'u_current' in point and 'v_current' in point:
                    u = float(point['u_current'].sel(depth=0).values)
                    v = float(point['v_current'].sel(depth=0).values)
                    speed = np.sqrt(u**2 + v**2)
                    direction = np.degrees(np.arctan2(v, u))
                    response += f"Current speed: {speed:.2f} m/s, Direction: {direction:.1f}°"
                    return response
            
            # If no specific variable, give overview
            response += "\n"
            if 'temperature' in point:
                temp = float(point['temperature'].sel(depth=0).values)
                response += f"- Temperature: {temp:.2f}°C\n"
            if 'salinity' in point:
                sal = float(point['salinity'].sel(depth=0).values)
                response += f"- Salinity: {sal:.2f} PSU\n"
            if 'chlorophyll' in point:
                chl = float(point['chlorophyll'].values)
                response += f"- Chlorophyll: {chl:.3f} mg/m³\n"
            
            return response.strip()
        
        # Handle aggregate queries
        if 'average' in question.lower() or 'mean' in question.lower():
            if 'temperature' in question.lower():
                avg_temp = float(ds['sea_surface_temperature'].mean().values)
                return f"The average sea surface temperature across all data is {avg_temp:.2f}°C"
            elif 'salinity' in question.lower():
                avg_sal = float(ds['salinity'].sel(depth=0).mean().values)
                return f"The average surface salinity across all data is {avg_sal:.2f} PSU"
        
        if 'maximum' in question.lower() or 'max' in question.lower() or 'highest' in question.lower():
            if 'temperature' in question.lower():
                max_temp = float(ds['sea_surface_temperature'].max().values)
                max_loc = ds['sea_surface_temperature'].where(
                    ds['sea_surface_temperature'] == ds['sea_surface_temperature'].max(), drop=True
                )
                max_time = pd.Timestamp(max_loc.time.values[0]).strftime('%Y-%m-%d')
                return f"The maximum sea surface temperature is {max_temp:.2f}°C, occurring on {max_time}"
        
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

    Improved, directive prompt structure for better RAG performance.
    """
    # ----------------------------------------------------------------------
    # PROMPT for Seq2Seq LM (Flan-T5): Designed to constrain output
    # ----------------------------------------------------------------------
    prompt = (
f"""Based ONLY on the following ocean data points, answer the user's question concisely and factually. 
If the information is not present in the data, state that you cannot answer based on the context.

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

        # Flan-T5 (Seq2Seq) generation parameters
        result = generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            # return_full_text is ignored for Seq2Seq pipelines, 
            # as they only return the generated answer sequence.
            truncation=True,
        )

        if result and len(result) > 0 and "generated_text" in result[0]:
            # The result should be the direct answer
            answer = result[0]["generated_text"].strip()
            
            # Since Flan-T5 is much better at constraint, we don't need complex cleaning
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