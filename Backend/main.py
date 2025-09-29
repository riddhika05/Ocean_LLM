from fastapi import FastAPI
from pydantic import BaseModel
import xarray as xr
import pandas as pd
from vector_store import VectorStore, embed_text
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
app = FastAPI()


# ----- Read NetCDF & flatten to text -----
ds = xr.open_dataset("data/ocean_data.nc")
df = ds.to_dataframe().reset_index()  # convert to flat table

texts = []
for _, row in df.iterrows():
    texts.append(f"Lat={row['lat']}, Lon={row['lon']}, Temp={row.get('temperature', 'NA')}C, Salinity={row.get('salinity', 'NA')}ppt")

# ----- Create vector store -----
vector_dim = 1536  # embedding size for text-embedding-3-small
store = VectorStore(dimension=vector_dim)

for text in texts[:100]:  # for demo, limit to first 100 rows
    emb = embed_text(text)
    store.add(text, emb)

# ----- API -----
class QueryRequest(BaseModel):
    question: str

def ask_llm(context, question):
    prompt = f"Answer the question using the following data:\n{context}\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@app.post("/query")
def query(request: QueryRequest):
    query_emb = embed_text(request.question)
    context = store.query(query_emb, top_k=1)[0]
    answer = ask_llm(context, request.question)
    return {"answer": answer}
