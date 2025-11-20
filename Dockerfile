# Dockerfile – IDS FastAPI service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside container
WORKDIR /app

# System deps (build + network libs if needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]" fastapi

# Copy project code
COPY . .

# Expose API port
EXPOSE 8000

# Default command: run FastAPI app
# Make sure src/api_main.py defines: app = FastAPI(...)
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]