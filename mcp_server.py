#!/usr/bin/env python3
"""
MCP (Model Control Protocol) Server for N8N Workflow Search
Exposes workflow search functionality as MCP tools.
"""

import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from workflow_db import WorkflowDatabase

# Initialize FastAPI app
app = FastAPI(
    title="N8N Workflow MCP Server",
    description="MCP server exposing workflow search functionality as tools",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = WorkflowDatabase()

# MCP Models
class MCPServerInfo(BaseModel):
    name: str = "n8n-workflow-search"
    version: str = "1.0.0"
    description: str = "MCP server for searching N8N workflows"

class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPResource(BaseModel):
    name: str
    description: str
    type: str
    tools: List[MCPTool]

class MCPListResponse(BaseModel):
    server: MCPServerInfo
    resources: List[MCPResource]

class SearchWorkflowsInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    threshold: float = Field(0.4, ge=0.0, le=1.0, description="Minimum similarity score (0-1)")

class SearchWorkflowsToolOutput(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total: int
    limit: int
    threshold: float

# MCP Endpoints
@app.get("/mcp/list", response_model=MCPListResponse)
async def list_resources():
    """List available MCP resources and tools."""
    return {
        "server": {
            "name": "n8n-workflow-search",
            "version": "1.0.0",
            "description": "MCP server for searching N8N workflows"
        },
        "resources": [
            {
                "name": "workflows",
                "description": "N8N workflow search and management",
                "type": "workflow",
                "tools": [
                    {
                        "name": "search_workflows",
                        "description": "Search workflows using semantic similarity",
                        "input_schema": SearchWorkflowsInput.schema()
                    }
                ]
            }
        ]
    }

@app.post("/mcp/workflows/search_workflows")
async def search_workflows_tool(input: SearchWorkflowsInput):
    """MCP tool for searching workflows using semantic similarity."""
    try:
        # Perform semantic search
        results = db.search_workflows_semantic(
            query=input.query,
            limit=input.limit,
            threshold=input.threshold
        )
        
        # Prepare response
        return {
            "results": results,
            "query": input.query,
            "total": len(results),
            "limit": input.limit,
            "threshold": input.threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

# Standard API Endpoints (for testing)
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    return db.get_stats()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='N8N Workflow MCP Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    print(f"Starting MCP server on http://{args.host}:{args.port}")
    print(f"- MCP endpoint: http://{args.host}:{args.port}/mcp")
    print(f"- API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "mcp_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
