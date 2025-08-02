# HackRX LLM-Powered Intelligent Query-Retrieval System
# Optimized for Railway deployment with enhanced accuracy features

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for Railway optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY .env* ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hackrx && \
    chown -R hackrx:hackrx /app
USER hackrx

# Expose port (Railway will set PORT env var)
EXPOSE 8080

# Health check (use Railway's PORT with longer timeout)
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=5 \
    CMD python -c "import requests, os; requests.get(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health', timeout=10)" || exit 1

# Start the application
CMD ["python", "main.py"]