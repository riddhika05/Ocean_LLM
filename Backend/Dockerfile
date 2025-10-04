# Stage 1: Build Stage
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for xarray, netCDF4, and possibly FAISS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libnetcdf-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
# Install dependencies, including google-genai, faiss-cpu, and dependencies for xarray/pandas
RUN pip install --no-cache-dir -r requirements.txt

# --- Optimizing for Deployment ---
# Removed pre-download steps for Hugging Face models (Sentence Transformer and Flan-T5) 
# as the application now relies entirely on the Gemini API for both embeddings and generation.
# --------------------------------

# Copy the rest of the application files
COPY . .

# Ensure the app starts on the correct port and host
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED 1

# Expose the port (default for uvicorn)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]