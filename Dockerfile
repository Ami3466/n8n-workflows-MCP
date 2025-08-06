# Use Python 3.9 slim as the base image
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Runtime stage
FROM python:3.9-slim

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/nltk_data /root/nltk_data

# Create appuser and set permissions
RUN useradd -m appuser && \
    mkdir -p /app /data && \
    chown -R appuser:appuser /app /data

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/workflows

# Set environment variables
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH="/app" \
    WORKFLOW_DB_PATH="/data/workflows.db" \
    PYTHONUNBUFFERED="1"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Command to run the application
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
