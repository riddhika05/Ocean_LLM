from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xarray as xr
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import re

# Import for Gemini API
from google import genai
from google.genai import types

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

# --- NOW Import heavy dependencies and Google GenAI SDK ---
# Assuming VectorStore and embed_text are available from vector_store
from vector_store import VectorStore, embed_text

# --- Initialize Gemini Client for Generation ---
print("Initializing Gemini Client...")
# Use the API Key from the environment variable 'GEMINI_API_KEY'
try:
    client = genai.Client()
    LLM_MODEL_NAME = "gemini-2.0-flash-exp"
    print(f"✓ Gemini Client initialized successfully with model: {LLM_MODEL_NAME}!")

except Exception as e:
    import traceback
    print(f"✗ Failed to initialize Gemini Client: {e}")
    print(traceback.format_exc())
    client = None
    LLM_MODEL_NAME = "none (failed)"


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

# Store dataset bounds for validation
DATASET_BOUNDS = None
if ds is not None:
    DATASET_BOUNDS = {
        'lat_min': float(ds.lat.min()),
        'lat_max': float(ds.lat.max()),
        'lon_min': float(ds.lon.min()),
        'lon_max': float(ds.lon.max())
    }
    print(f"Dataset bounds: Lat [{DATASET_BOUNDS['lat_min']}, {DATASET_BOUNDS['lat_max']}], "
          f"Lon [{DATASET_BOUNDS['lon_min']}, {DATASET_BOUNDS['lon_max']}]")

# ----- Create vector store with enhanced data chunks -----
vector_dim = 768  # for Gemini 'text-embedding-004'
store = VectorStore(dimension=vector_dim)
texts = []

if ds is not None:
    print("Preparing enhanced ocean data chunks for indexing...")
    
    # Sample spatial points and create descriptive summaries
    # Sample every 5th lat/lon point for better coverage
    lat_sample = ds.lat.values[::5]
    lon_sample = ds.lon.values[::5]
    
    # Sample times: all time steps (monthly already)
    time_sample = ds.time.values
    
    chunk_count = 0
    # Index more data points (5 lat × 5 lon × 12 months = 300 points)
    for lat in lat_sample[:5]:  # Use 5 lat points
        for lon in lon_sample[:5]:  # Use 5 lon points
            for time in time_sample[:12]:  # All 12 months
                try:
                    # Extract data at this point
                    point = ds.sel(lat=lat, lon=lon, time=time, method='nearest')
                    
                    # Create descriptive text chunk (NO special characters)
                    text = (
                        f"Location: Latitude {float(lat):.2f} degrees North, Longitude {float(lon):.2f} degrees East. "
                        f"Date: {pd.Timestamp(time).strftime('%Y-%m-%d')}. "
                    )
                    
                    # Add temperature
                    if 'temperature' in ds:
                        temp = float(point['temperature'].values)
                        text += f"Sea surface temperature: {temp:.2f} degrees Celsius. "
                    
                    # Add salinity
                    if 'salinity' in ds:
                        sal = float(point['salinity'].values)
                        text += f"Surface salinity: {sal:.2f} PSU. "
                    
                    texts.append(text)
                    chunk_count += 1
                    
                except Exception as e:
                    print(f"Error creating chunk: {e}")
                    continue
    
    print(f"Created {len(texts)} descriptive data chunks")
    
    # Index all chunks
    if texts:
        print(f"Indexing {len(texts)} ocean data points...")
        for i, text in enumerate(texts):
            try:
                emb = embed_text(text)
                store.add(text, emb)
                if (i + 1) % 50 == 0:
                    print(f"  Indexed {i + 1}/{len(texts)} chunks...")
            except Exception as e:
                print(f"Error embedding text chunk {i}: {e}")
                # Continue with other chunks
                continue
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


def check_coordinates_in_bounds(lat, lon):
    """Check if coordinates are within dataset bounds."""
    if DATASET_BOUNDS is None:
        return True, None
    
    in_bounds = (
        DATASET_BOUNDS['lat_min'] <= lat <= DATASET_BOUNDS['lat_max'] and
        DATASET_BOUNDS['lon_min'] <= lon <= DATASET_BOUNDS['lon_max']
    )
    
    if not in_bounds:
        message = (
            f"Coordinates (lat={lat}, lon={lon}) are outside the dataset coverage area. "
            f"Dataset covers: Latitude {DATASET_BOUNDS['lat_min']:.1f} to {DATASET_BOUNDS['lat_max']:.1f}°N, "
            f"Longitude {DATASET_BOUNDS['lon_min']:.1f} to {DATASET_BOUNDS['lon_max']:.1f}°E."
        )
        return False, message
    
    return True, None


def query_dataset_directly(question: str, params: dict):
    """Directly query the xarray dataset for specific point queries."""
    if ds is None:
        return None
    
    try:
        # Handle aggregate queries FIRST (before coordinate queries)
        if 'average' in question.lower() or 'mean' in question.lower():
            if 'temperature' in question.lower():
                avg_temp = float(ds['temperature'].mean().values)
                return f"The average sea surface temperature across all data is {avg_temp:.2f} degrees Celsius"
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
                max_lat = float(max_loc.lat.values[0])
                max_lon = float(max_loc.lon.values[0])
                return (f"The maximum sea surface temperature is {max_temp:.2f} degrees Celsius, "
                       f"occurring at lat={max_lat:.2f}°N, lon={max_lon:.2f}°E on {max_time}")
            elif 'salinity' in question.lower():
                max_sal = float(ds['salinity'].max().values)
                return f"The maximum surface salinity is {max_sal:.2f} PSU"
        
        if 'minimum' in question.lower() or 'min' in question.lower() or 'lowest' in question.lower():
            if 'temperature' in question.lower():
                min_temp = float(ds['temperature'].min().values)
                min_loc = ds['temperature'].where(
                    ds['temperature'] == ds['temperature'].min(), drop=True
                )
                min_time = pd.Timestamp(min_loc.time.values[0]).strftime('%Y-%m-%d')
                min_lat = float(min_loc.lat.values[0])
                min_lon = float(min_loc.lon.values[0])
                return (f"The minimum sea surface temperature is {min_temp:.2f} degrees Celsius, "
                       f"occurring at lat={min_lat:.2f}°N, lon={min_lon:.2f}°E on {min_time}")
            elif 'salinity' in question.lower():
                min_sal = float(ds['salinity'].min().values)
                return f"The minimum surface salinity is {min_sal:.2f} PSU"
        
        # If we have lat/lon, try direct query
        if 'lat' in params and 'lon' in params:
            lat, lon = params['lat'], params['lon']
            
            # Check if coordinates are in bounds
            in_bounds, message = check_coordinates_in_bounds(lat, lon)
            if not in_bounds:
                return message
            
            # Select point using nearest neighbor (this ALWAYS works for in-bounds coords)
            point = ds.sel(lat=lat, lon=lon, method='nearest')
            
            # Get the actual coordinates that were selected
            actual_lat = float(point.lat.values)
            actual_lon = float(point.lon.values)
            
            # Add time if specified
            if 'time' in params:
                point = point.sel(time=params['time'], method='nearest')
                time_str = params['time']
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
                    time_str = params['month']
                else:
                    time_str = "all available times"
            else:
                # Use the first time point if no time specified
                point = point.isel(time=0)
                time_str = pd.Timestamp(point.time.values).strftime('%Y-%m')
            
            # Build response
            response = f"At location lat={actual_lat:.2f}°N, lon={actual_lon:.2f}°E"
            if actual_lat != lat or actual_lon != lon:
                response += f" (nearest to requested {lat}°N, {lon}°E)"
            if time_str != "all available times":
                response += f" in {time_str}"
            response += ":\n"
            
            # Check what variable is being asked about
            if 'temperature' in question.lower() and 'temperature' in point:
                temp = float(point['temperature'].values)
                response += f"Temperature: {temp:.2f} degrees Celsius"
                return response
            
            elif 'salinity' in question.lower() and 'salinity' in point:
                sal = float(point['salinity'].values)
                response += f"Salinity: {sal:.2f} PSU"
                return response
            
            # If no specific variable mentioned, give overview
            result_parts = []
            if 'temperature' in point:
                temp = float(point['temperature'].values)
                result_parts.append(f"Temperature: {temp:.2f} degrees Celsius")
            if 'salinity' in point:
                sal = float(point['salinity'].values)
                result_parts.append(f"Salinity: {sal:.2f} PSU")
            
            if result_parts:
                response += "\n".join(result_parts)
                return response.strip()
        
    except Exception as e:
        print(f"Error in direct query: {e}")
        import traceback
        print(traceback.format_exc())
        return None
    
    return None


# ----- API Schemas -----
class QueryRequest(BaseModel):
    question: str


def ask_llm(context: str, question: str) -> str:
    """
    Generate answer using the Gemini model based on retrieved context.
    """
    if client is None:
        return "Error: Gemini Client failed to initialize. Cannot generate response."

    # Add dataset bounds to context if available
    bounds_info = ""
    if DATASET_BOUNDS:
        bounds_info = f"\nDataset Coverage: Latitude {DATASET_BOUNDS['lat_min']:.1f}-{DATASET_BOUNDS['lat_max']:.1f}°N, Longitude {DATASET_BOUNDS['lon_min']:.1f}-{DATASET_BOUNDS['lon_max']:.1f}°E\n"

    prompt = f"""You are an expert Oceanographer AI assistant. Answer the user's question based ONLY on the provided data context below.
{bounds_info}
Data Context:
{context}

User Question: {question}

Instructions:
- Provide a clear, concise answer based on the data
- If the data doesn't contain the answer, say so explicitly
- If coordinates are mentioned that are outside the dataset coverage, state this clearly
- Use proper units (degrees Celsius for temperature, PSU for salinity)
- Be specific with numbers when available

Answer:"""

    try:
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=500,
        )

        response = client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=prompt,
            config=config,
        )
        
        # Check if response.text is None
        if response.text is None:
            # Check for safety blocking
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason'):
                    return f"Response blocked by safety filters: {response.prompt_feedback.block_reason}"
            
            # Check if candidates were generated
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    return f"Generation stopped: {candidate.finish_reason}"
            
            return "No text was generated by the model. Please try rephrasing your question."

        answer = response.text.strip()
        return answer if answer else "No answer could be generated. Please try a different question."

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error generating response: {str(e)}"


@app.post("/query")
def query(request: QueryRequest):
    """Handle query requests with RAG pipeline."""
    try:
        print(f"\n--- Processing Query: {request.question} ---")
        
        # Step 1: Try direct dataset query first (for specific point queries)
        params = extract_query_params(request.question)
        direct_answer = query_dataset_directly(request.question, params)
        
        if direct_answer:
            print("✓ Answered using direct query")
            return {
                "answer": direct_answer,
                "method": "direct_query",
                "context_used": [],
                "coordinates": params if params else None
            }
        
        # Step 2: Fall back to RAG pipeline
        print("Using RAG pipeline...")
        query_emb = embed_text(request.question)
        contexts = store.query(query_emb, top_k=5)
        
        print(f"Retrieved {len(contexts)} context chunks")
        
        combined_context = "\n\n".join(contexts)

        if not combined_context or len(contexts) == 0:
            answer = "No relevant data found in the vector store to answer your question. Please try a different query or check if the dataset is properly loaded."
        else:
            print("Generating answer with LLM...")
            answer = ask_llm(combined_context, request.question)

        return {
            "answer": answer,
            "method": "rag_pipeline",
            "context_used": contexts,
            "num_contexts": len(contexts)
        }
    except Exception as e:
        import traceback
        print(f"Error in query endpoint: {e}")
        print(traceback.format_exc())
        return {
            "answer": f"Error processing query: {str(e)}",
            "context_used": [],
            "error": str(e)
        }


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Bay of Bengal Ocean RAG API is running (powered by Gemini)",
        "dataset_bounds": DATASET_BOUNDS,
        "endpoints": {
            "POST /query": "Submit a question about ocean data",
            "GET /health": "Check API health",
            "GET /dataset/info": "Get dataset information"
        }
    }


@app.get("/health")
def health():
    # Model status check
    model_status = "loaded" if client else "failed to load"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "model_name": LLM_MODEL_NAME,
        "data_points": len(texts),
        "vector_store_size": store.index.ntotal if store else 0,
        "dataset_loaded": ds is not None,
        "dataset_variables": list(ds.data_vars) if ds is not None else [],
        "dataset_path": DATA_PATH if ds is not None else "not found",
        "dataset_bounds": DATASET_BOUNDS
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
        "attributes": dict(ds.attrs),
        "bounds": DATASET_BOUNDS
    }