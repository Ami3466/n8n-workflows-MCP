#!/usr/bin/env python3
"""
MCP (Model Control Protocol) Server for N8N Workflow Search
Exposes workflow search functionality as MCP tools.
"""

import os
import json
import uvicorn
import json
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Models
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int, None]]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

# In-memory storage for workflows
workflows = [
    {"id": "1", "name": "Example Workflow", "description": "A sample workflow"},
    {"id": "2", "name": "Data Processing", "description": "Process data from API"},
]

# MCP Handlers
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle MCP requests and route to appropriate handler"""
    try:
        if not request.method:
            raise ValueError("No method specified")
            
        if request.method == "initialize":
            return handle_initialize()
        elif request.method == "tools/list":
            return handle_list_tools()
        elif request.method == "tools/call":
            return await handle_call_tool(request.params or {})
        else:
            return MCPResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {request.method}"}
            )
    except Exception as e:
        logger.exception("Error handling MCP request")
        return MCPResponse(
            id=request.id,
            error={"code": -32603, "message": f"Internal error: {str(e)}"}
        )

def handle_initialize() -> MCPResponse:
    """Handle MCP initialize request"""
    return MCPResponse(
        id=None,
        result={
            "serverInfo": {
                "name": "N8N Workflows MCP",
                "version": "0.1.0",
                "description": "MCP server for managing and searching N8N workflows"
            },
            "capabilities": {
                "tools": {"enabled": True}
            }
        }
    )

def handle_list_tools() -> MCPResponse:
    """Handle tools/list request"""
    return MCPResponse(
        id=None,
        result={
            "tools": [
                {
                    "name": "search_workflows",
                    "description": "Search for workflows by name or description",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "minimum": 1, "default": 5}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_workflow",
                    "description": "Get a workflow by ID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "workflow_id": {"type": "string", "description": "Workflow ID"}
                        },
                        "required": ["workflow_id"]
                    }
                }
            ]
        }
    )

async def handle_call_tool(params: Dict[str, Any]) -> MCPResponse:
    """Handle tools/call request"""
    tool_name = params.get("name")
    tool_params = params.get("parameters", {})
    
    if tool_name == "search_workflows":
        query = tool_params.get("query", "").lower()
        limit = min(int(tool_params.get("limit", 5)), 20)
        
        results = [
            wf for wf in workflows 
            if query in wf["name"].lower() or query in wf["description"].lower()
        ][:limit]
        
        return MCPResponse(
            id=params.get("id"),
            result={"results": results}
        )
        
    elif tool_name == "get_workflow":
        workflow_id = tool_params.get("workflow_id")
        for wf in workflows:
            if wf["id"] == workflow_id:
                return MCPResponse(
                    id=params.get("id"),
                    result={"workflow": wf}
                )
        
        return MCPResponse(
            id=params.get("id"),
            error={"code": 404, "message": "Workflow not found"}
        )
    
    return MCPResponse(
        id=params.get("id"),
        error={"code": 404, "message": f"Tool not found: {tool_name}"}
    )

# HTTP Endpoints
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint that handles JSON-RPC requests"""
    try:
        # Parse request body
        try:
            body = await request.json()
            mcp_request = MCPRequest(**body)
        except Exception as e:
            logger.error(f"Invalid request: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid request format")
        
        # Handle the MCP request
        response = await handle_mcp_request(mcp_request)
        
        # Set the ID from request if not set in response
        if response.id is None:
            response.id = mcp_request.id
            
        return response.dict(exclude_none=True)
        
    except Exception as e:
        logger.exception("Error processing MCP request")
        return MCPResponse(
            id=getattr(mcp_request, 'id', None),
            error={"code": -32603, "message": f"Internal error: {str(e)}"}
        ).dict(exclude_none=True)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "n8n-workflows-mcp"
    }

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
