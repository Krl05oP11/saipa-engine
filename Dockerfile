FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for chromadb and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8052

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8052"]
