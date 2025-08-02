#!/usr/bin/env python3
"""
Startup script for Bajaj HackRX Azure Application
Handles environment setup and graceful startup
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_SEARCH_ENDPOINT', 
        'AZURE_SEARCH_API_KEY',
        'AZURE_SQL_SERVER',
        'AZURE_SQL_DATABASE',
        'AZURE_SQL_USERNAME',
        'AZURE_SQL_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Running in demo mode with mock responses")
        return False
    
    return True

def setup_demo_environment():
    """Set up demo environment variables if Azure services are not configured"""
    demo_vars = {
        'AZURE_OPENAI_ENDPOINT': 'https://demo.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'demo-key',
        'AZURE_SEARCH_ENDPOINT': 'https://demo.search.windows.net',
        'AZURE_SEARCH_API_KEY': 'demo-search-key',
        'AZURE_SQL_SERVER': 'demo.database.windows.net',
        'AZURE_SQL_DATABASE': 'demo-db',
        'AZURE_SQL_USERNAME': 'demo-user',
        'AZURE_SQL_PASSWORD': 'demo-password'
    }
    
    for key, value in demo_vars.items():
        if not os.getenv(key):
            os.environ[key] = value

def main():
    """Main startup function"""
    logger.info("Starting Bajaj HackRX Azure Application...")
    
    # Check environment
    has_azure_config = check_environment()
    
    if not has_azure_config:
        setup_demo_environment()
        logger.info("Demo environment configured")
    else:
        logger.info("Azure environment validated")
    
    # Import and run the application
    try:
        import uvicorn
        from bajajhackrx import app
        
        port = int(os.environ.get('PORT', 8000))
        
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Running fallback server...")
        
        # Fallback to simple server if dependencies fail
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class FallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "message": "Bajaj HackRX API - Fallback Mode",
                    "status": "dependencies_missing",
                    "note": "Some Azure dependencies are not available"
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
        
        server = HTTPServer(('', port), FallbackHandler)
        logger.info(f"Fallback server running on port {port}")
        server.serve_forever()
    
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
