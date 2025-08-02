# HackRX LLM-Powered Intelligent Query-Retrieval System
# Optimized for Railway deployment with enhanced accuracy features

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

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

# Health check (use Railway's PORT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests, os; requests.get(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

# Start the application
CMD ["python", "main.py"]