FROM python:3.11-slim

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# make src importable
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["bash", "-lc", "uvicorn src.api_main:app --host 0.0.0.0 --port ${PORT:-8000}"]rn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]