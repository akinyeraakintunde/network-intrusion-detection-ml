FROM python:3.11-slim

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Make /app importable so "src" can be found
ENV PYTHONPATH=/app

# Render provides $PORT
CMD uvicorn src.api_main:app --host 0.0.0.0 --port ${PORT:-10000}