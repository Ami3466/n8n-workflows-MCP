# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKFLOW_DB_PATH="/data/workflows.db" \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set work directory
WORKDIR /app

# Install system dependencies
RUN set -ex \
    && BUILD_DEPS=" \
        gcc \
        python3-dev \
    " \
    && apt-get update \
    && apt-get install -y --no-install-recommends $BUILD_DEPS \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ -n "$BUILD_DEPS" ]; then \
        apt-get purge -y --auto-remove $BUILD_DEPS; \
       fi \
    && rm -rf /root/.cache/pip \
    && find /usr/local -depth \
        \( \
            \( -type d -a \( -name test -o -name tests -o -name '__pycache__' \) \) \
            -o \
            \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
        \) -exec rm -rf '{}' +

# Copy application code
COPY mcp_server.py workflow_db.py ./

# Create data directory with proper permissions
RUN mkdir -p /data \
    && chmod 777 /data

# Create a non-root user and switch to it
RUN groupadd -r appuser \
    && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app /data
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Command to run the application
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
