FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# IMPORTANT: make sure Python can import "src"
ENV PYTHONPATH=/app

# Render provides PORT
CMD sh -c "uvicorn src.api_main:app --host 0.0.0.0 --port ${PORT:-8000}"