#!/usr/bin/env python3
"""
Initialize the workflow database and download required NLTK data.
"""
import os
import nltk
from workflow_db import WorkflowDatabase

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK data downloaded successfully.")

def init_database():
    """Initialize the workflow database."""
    print("Initializing workflow database...")
    db = WorkflowDatabase()
    print("Database initialized successfully.")
    return db

if __name__ == "__main__":
    # Download required NLTK data
    download_nltk_data()
    
    # Initialize database
    db = init_database()
    
    print("\nSetup complete! You can now start the MCP server with:")
    print("  uvicorn mcp_server:app --reload")
    print("\nOr using Docker Compose:")
    print("  docker-compose up --build")
