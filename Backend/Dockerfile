# Stage 1: Build Stage - Use a slightly heavier base image that often 
# accelerates PyTorch/ML dependency installation and avoids runtime errors.
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for xarray, netCDF4, and possibly FAISS)
# We also install Git to manage the large model files downloaded by Hugging Face
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libnetcdf-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
# This is slow, but Docker caches this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Optimizing for Deployment ---
# Hugging Face models are huge. We pre-download the model weights during the build 
# step to prevent a timeout when the app tries to load the model on startup.

# 1. Download Sentence Transformer Embedding Model (all-MiniLM-L6-v2)
ENV EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\"${EMBEDDING_MODEL_NAME}\")"

# 2. Download Flan-T5 LLM Model (google/flan-t5-base)
ENV LLM_MODEL_NAME="google/flan-t5-base"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained(\"${LLM_MODEL_NAME}\"); AutoModelForSeq2SeqLM.from_pretrained(\"${LLM_MODEL_NAME}\")"
# --------------------------------

# Copy the rest of the application files
COPY . .

# Ensure the app starts on the correct port and host
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED 1

# Expose the port (default for uvicorn)
EXPOSE 8000

# Command to run the application (equivalent to your Procfile)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]