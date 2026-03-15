FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch + torchaudio + other dependencies
RUN pip install --no-cache-dir \
    torch==2.10.0+cpu \
    torchaudio==2.10.0+cpu \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy backend code
COPY backend/ .

# Create directories for uploads/models
RUN mkdir -p uploads models

# Optional: remove torch tests/docs (~200MB)
RUN find /usr/local/lib/python3.10/site-packages/torch -name "tests" -type d -exec rm -rf {} + \
 && find /usr/local/lib/python3.10/site-packages/torch -name "docs" -type d -exec rm -rf {} +

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]