# Minimal Railway Dockerfile for HackRX
FROM python:3.11-slim

WORKDIR /app

# Essential environment variables only
ENV PYTHONUNBUFFERED=1

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY .env* ./

# Simple port exposure
EXPOSE 8080

# Minimal health check
HEALTHCHECK --interval=300s --timeout=30s --start-period=30s --retries=2 \
    CMD python -c "import requests, os; requests.get(f'http://localhost:{os.environ.get(\"PORT\", 8080)}/health', timeout=5)" || exit 1

# Start application
CMD ["python", "main.py"]