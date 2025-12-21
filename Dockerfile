FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but safe)
RUN apt-get update && apt-get install -y build-essential

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8000

# 🔑 IMPORTANT PART
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "."]