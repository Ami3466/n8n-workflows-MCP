# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKFLOW_DB_PATH="/data/workflows.db"

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with minimal cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY mcp_server.py .
COPY workflow_db.py .

# Create data directory
RUN mkdir -p /data

# Expose port 8000
EXPOSE 8000

# Simple health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Command to run the application
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
