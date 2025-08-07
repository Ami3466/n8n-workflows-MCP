# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only FastMCP and FastAPI
RUN pip install --no-cache-dir fastmcp==0.1.0 fastapi==0.104.0 uvicorn==0.24.0

# Copy only the necessary files
COPY mcp_server.py .

# Create a simple health check script
RUN echo '#!/bin/sh\ncurl -f http://localhost:8000/health || exit 1' > /healthcheck.sh \
    && chmod +x /healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Command to run the application
CMD ["uvicorn", "mcp_server:mcp.app", "--host", "0.0.0.0", "--port", "8000"]
