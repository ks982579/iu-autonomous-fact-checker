# ./run_api.py
"""
Startup script for the Fact Checking API
Run this from the project root directory
"""
import json
import uvicorn
from pathlib import Path


def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: config.json not found at {config_path}, using defaults")
        return {
            "api": {
                "host": "localhost", 
                "port": 8000
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error reading config.json: {e}, using defaults")
        return {
            "api": {
                "host": "localhost", 
                "port": 8000
            }
        }


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    api_config = config.get("api", {})
    
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    
    print(f"Starting Fact Checking API on {host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    
    # Run the FastAPI application
    uvicorn.run(
        "api.main:app", 
        host=host, 
        port=port, 
        reload=True,  # Enable hot-reload
        log_level="info"
    )