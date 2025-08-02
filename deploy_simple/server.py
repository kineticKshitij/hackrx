#!/usr/bin/env python3
"""
Simple HTTP server for Bajaj HackRX API demo
Minimal dependencies for Azure deployment
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

class BajajHackRXHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "message": "Welcome to Bajaj HackRX API",
                "status": "success",
                "version": "1.0.0",
                "endpoints": [
                    "/health - Health check",
                    "/query - Document query endpoint",
                    "/search - Search documents",
                    "/analyze - Analyze data"
                ]
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "service": "Bajaj HackRX API"}
            self.wfile.write(json.dumps(response).encode())
            
        elif parsed_path.path == '/query':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            query_params = parse_qs(parsed_path.query)
            user_query = query_params.get('q', [''])[0]
            
            response = {
                "query": user_query or "No query provided",
                "result": "This is a demo response for the Bajaj HackRX competition",
                "status": "success",
                "data": {
                    "document_matches": 3,
                    "confidence_score": 0.85,
                    "suggested_actions": [
                        "Review document section 2.1",
                        "Check compliance requirements",
                        "Verify with legal team"
                    ]
                }
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif parsed_path.path == '/search':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "search_results": [
                    {"id": 1, "title": "Financial Policy Document", "relevance": 0.92},
                    {"id": 2, "title": "Compliance Guidelines", "relevance": 0.87},
                    {"id": 3, "title": "Risk Assessment Report", "relevance": 0.78}
                ],
                "total_results": 3,
                "status": "success"
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif parsed_path.path == '/analyze':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "analysis": {
                    "total_documents": 1250,
                    "processed_today": 47,
                    "avg_processing_time": "2.3s",
                    "accuracy_rate": "94.2%"
                },
                "insights": [
                    "Peak usage between 9-11 AM",
                    "Most queries related to financial documents",
                    "High accuracy on structured documents"
                ],
                "status": "success"
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Endpoint not found", "status": "error"}
            self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        if self.path == '/query':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                query = data.get('query', '')
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    "query": query,
                    "result": f"Processed query: {query}",
                    "status": "success",
                    "processing_time": "1.2s",
                    "confidence": 0.89
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"error": "Invalid JSON", "status": "error"}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(405)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Method not allowed", "status": "error"}
            self.wfile.write(json.dumps(response).encode())

def run_server():
    port = int(os.environ.get('PORT', 8000))
    server_address = ('', port)
    httpd = HTTPServer(server_address, BajajHackRXHandler)
    print(f"Starting Bajaj HackRX API server on port {port}")
    print(f"Visit http://localhost:{port} to access the API")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
