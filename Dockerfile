# Use slim Python 3.10 image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by torchaudio (libsndfile, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — reinstall only if deps change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source files
COPY backend/ .

# Create uploads directory
RUN mkdir -p uploads models 

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]