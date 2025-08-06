#!/usr/bin/env python3
"""
Startup script for the Fact Checking API
Run this from the project root directory
"""
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )