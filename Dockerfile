# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Copy the test application
COPY test_app.py .

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "test_app:app", "--host", "0.0.0.0", "--port", "8000"]
