FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source + data
COPY src ./src
COPY data ./data

# Make "src" importable as a package
ENV PYTHONPATH=/app

# Render provides PORT
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api_main:app --host 0.0.0.0 --port ${PORT:-8000}"]