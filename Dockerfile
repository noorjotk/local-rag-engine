# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# PRE-DOWNLOAD THE EMBEDDING MODEL (This fixes the 3-4 minute delay!)
RUN python -c "from sentence_transformers import SentenceTransformer; print('🔄 Downloading BAAI/bge-large-en-v1.5...'); model = SentenceTransformer('BAAI/bge-large-en-v1.5'); print('✅ Model downloaded successfully')"


# Copy all code
COPY . .

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface


# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Start uvicorn and Streamlit together using a simple script
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]

# This says:
#     “Build me a Python image, put my FastAPI code in it, install dependencies, expose port 8000, run FastAPI.”

# This means:

#     One container runs both FastAPI + Streamlit.
#     The CMD runs both processes — background uvicorn & Streamlit.
    

#uvicorn app:app (filename:app)  means first app is taken from filename app.py 