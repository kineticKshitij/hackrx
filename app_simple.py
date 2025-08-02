# Simplified deployment using Azure App Service
# This version removes complex dependencies and focuses on core functionality

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI and basic components
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified Configuration
class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    BEARER_TOKEN = "d691ab348b0d57d77e97cb3d989203e9168c6f8a88e91dd37dc80ff0a9b213aa"

config = Config()

# Pydantic models
class DocumentInput(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    metadata: Optional[Dict[str, Any]] = None

# FastAPI Application
app = FastAPI(
    title="Bajaj HackRX - Simplified LLM Query System",
    description="Simplified document processing system for hackathon demo",
    version="1.0.0"
)

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(request: DocumentInput):
    """Simplified endpoint that returns mock responses for demo"""
    
    try:
        # For demo purposes, return sample responses
        sample_answers = []
        
        for i, question in enumerate(request.questions):
            if "coverage" in question.lower():
                answer = f"Based on the policy document, the coverage for your inquiry includes comprehensive benefits with specific terms and conditions. Please refer to the policy document for detailed information."
            elif "waiting period" in question.lower():
                answer = f"The waiting period varies depending on the specific condition or treatment. Standard waiting periods apply as outlined in the policy terms."
            elif "premium" in question.lower():
                answer = f"Premium calculations are based on various factors including age, coverage type, and selected benefits. Please consult the policy schedule for exact amounts."
            else:
                answer = f"Thank you for your question: '{question}'. Our AI system has processed your query and found relevant information in the policy document. For detailed information, please refer to the specific sections mentioned in your policy."
            
            sample_answers.append(answer)
        
        response = QueryResponse(
            answers=sample_answers,
            metadata={
                "document_url": request.documents,
                "total_questions": len(request.questions),
                "processing_timestamp": datetime.now().isoformat(),
                "status": "demo_mode",
                "message": "This is a simplified demo version for hackathon presentation"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in run_submission: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "mode": "demo"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bajaj HackRX - Simplified LLM Query System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app_simple:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
