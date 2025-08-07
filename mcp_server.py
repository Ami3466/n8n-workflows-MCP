#!/usr/bin/env python3
"""
MCP (Model Control Protocol) Server for N8N Workflow Search
Exposes workflow search functionality as MCP tools.
"""

import os
import json
import uvicorn
from fastmcp import FastMCP, MCPResult
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import time

# Initialize MCP server
mcp = FastMCP(
    name="N8N Workflows MCP",
    description="MCP server for managing and searching N8N workflows",
    version="0.1.0"
)

# Simple in-memory storage for testing
workflows = [
    {"id": "1", "name": "Example Workflow", "description": "A sample workflow"},
    {"id": "2", "name": "Data Processing", "description": "Process data from API"},
]

# Define models
class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Max number of results")

class Workflow(BaseModel):
    id: str
    name: str
    description: str

# Register tools
@mcp.tool(
    name="search_workflows",
    description="Search for workflows",
    input_model=SearchParams,
    output_model=List[Workflow]
)
async def search_workflows(params: SearchParams) -> MCPResult:
    """Search for workflows matching the query"""
    try:
        query = params.query.lower()
        results = [
            wf for wf in workflows 
            if query in wf["name"].lower() or query in wf["description"].lower()
        ][:params.limit]
        return MCPResult(content=results, success=True)
    except Exception as e:
        return MCPResult(content={"error": str(e)}, success=False)

@mcp.tool(
    name="get_workflow",
    description="Get a workflow by ID",
    input_model=dict,
    output_model=Workflow
)
async def get_workflow(params: dict) -> MCPResult:
    """Get a workflow by its ID"""
    try:
        workflow_id = params.get("workflow_id")
        for wf in workflows:
            if wf["id"] == workflow_id:
                return MCPResult(content=wf, success=True)
        return MCPResult(content={"error": "Workflow not found"}, success=False)
    except Exception as e:
        return MCPResult(content={"error": str(e)}, success=False)

# Health check endpoint
@mcp.app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Run the server
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
