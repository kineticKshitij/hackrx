# Bajaj HackRX - Azure LLM Intelligent Query-Retrieval System

## Overview
This is an Azure-powered intelligent document processing and query-retrieval system that uses Azure OpenAI, Azure Cognitive Search, and Azure SQL Database.

## Features
- 📄 Multi-format document processing (PDF, DOCX, EML, TXT)
- 🔍 Vector search with Azure Cognitive Search
- 🤖 GPT-4 powered query understanding
- 💾 Azure SQL Database for analytics
- 🔒 Enterprise-grade security
- 📊 Performance metrics and monitoring

## Architecture
```
FastAPI Application
├── Document Processing (PyPDF2, python-docx)
├── Text Chunking & Embeddings (Azure OpenAI)
├── Vector Search (Azure Cognitive Search)
├── Query Processing (Azure OpenAI GPT-4)
└── Analytics Storage (Azure SQL Database)
```

## Prerequisites

### Azure Services Required
1. **Azure OpenAI Service**
   - GPT-4 deployment
   - text-embedding-ada-002 deployment

2. **Azure Cognitive Search**
   - Standard tier or higher (for vector search)

3. **Azure SQL Database**
   - Basic tier or higher

### Development Requirements
- Python 3.11+
- Docker (for containerization)
- Azure CLI (for deployment)

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd hackrx
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file or set environment variables:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_MODEL_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure Cognitive Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=document-chunks

# Azure SQL Database Configuration
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your-database
AZURE_SQL_USERNAME=your-username
AZURE_SQL_PASSWORD=your-password
```

### 3. Run Locally
```bash
python bajajhackrx.py
```

The API will be available at `http://localhost:8000`

## Deployment to Azure

### Option 1: Automated Deployment (Recommended)
```bash
# Linux/Mac
chmod +x deploy-azure.sh
./deploy-azure.sh

# Windows
deploy-azure.bat
```

### Option 2: Manual Deployment

#### Step 1: Create Azure Resources
```bash
# Login to Azure
az login

# Create resource group
az group create --name bajaj-hackrx-rg --location eastus

# Create Azure Container Registry
az acr create --resource-group bajaj-hackrx-rg --name bajajhackrxacr --sku Basic
```

#### Step 2: Build and Push Container
```bash
# Build and push to ACR
az acr build --registry bajajhackrxacr --image bajaj-hackrx-api:latest .
```

#### Step 3: Deploy to Container Apps
```bash
# Create Container App Environment
az containerapp env create \
    --name bajaj-hackrx-env \
    --resource-group bajaj-hackrx-rg \
    --location eastus

# Create Container App
az containerapp create \
    --name bajaj-hackrx-api \
    --resource-group bajaj-hackrx-rg \
    --environment bajaj-hackrx-env \
    --image bajajhackrxacr.azurecr.io/bajaj-hackrx-api:latest \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 10 \
    --cpu 1.0 \
    --memory 2.0Gi
```

#### Step 4: Configure Environment Variables
```bash
az containerapp update \
    --name bajaj-hackrx-api \
    --resource-group bajaj-hackrx-rg \
    --set-env-vars \
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/ \
        AZURE_OPENAI_API_KEY=your-key \
        AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net \
        AZURE_SEARCH_API_KEY=your-search-key \
        AZURE_SQL_SERVER=your-server.database.windows.net \
        AZURE_SQL_DATABASE=your-database \
        AZURE_SQL_USERNAME=your-username \
        AZURE_SQL_PASSWORD=your-password
```

## API Usage

### Authentication
All requests require a Bearer token in the Authorization header:
```
Authorization: Bearer d691ab348b0d57d77e97cb3d989203e9168c6f8a88e91dd37dc80ff0a9b213aa
```

### Main Endpoint
```http
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer d691ab348b0d57d77e97cb3d989203e9168c6f8a88e91dd37dc80ff0a9b213aa

{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the coverage for dental treatment?",
        "What is the waiting period for pre-existing conditions?"
    ]
}
```

### Response
```json
{
    "answers": [
        "Based on the policy document, dental treatment coverage...",
        "The waiting period for pre-existing conditions is..."
    ],
    "metadata": {
        "document_url": "https://example.com/document.pdf",
        "total_questions": 2,
        "processing_timestamp": "2025-08-02T10:30:00",
        "average_processing_time": 2.34,
        "azure_services_used": [
            "Azure OpenAI Service",
            "Azure Cognitive Search",
            "Azure SQL Database"
        ]
    }
}
```

### Other Endpoints
- `GET /health` - Health check with Azure services status
- `GET /metrics` - Performance metrics from Azure SQL
- `GET /docs` - Interactive API documentation

## Monitoring and Troubleshooting

### Health Check
```bash
curl https://your-app-url.azurecontainerapps.io/health
```

### View Logs
```bash
az containerapp logs show \
    --name bajaj-hackrx-api \
    --resource-group bajaj-hackrx-rg \
    --follow
```

### Performance Metrics
```bash
curl https://your-app-url.azurecontainerapps.io/metrics
```

## Cost Optimization

### Azure Service Tiers
- **Development**: Basic tiers for all services (~$50-100/month)
- **Production**: Standard tiers with auto-scaling (~$200-500/month)

### Scaling Configuration
```bash
# Update scaling rules
az containerapp update \
    --name bajaj-hackrx-api \
    --resource-group bajaj-hackrx-rg \
    --min-replicas 0 \
    --max-replicas 5
```

## Security Considerations

1. **Store secrets in Azure Key Vault**
2. **Use Managed Identity for Azure service authentication**
3. **Enable Azure Application Gateway for additional security**
4. **Implement request rate limiting**
5. **Use Azure Private Endpoints for database connections**

## Support

For issues and questions:
1. Check the `/health` endpoint for service status
2. Review logs using Azure CLI
3. Monitor performance metrics at `/metrics`
4. Check Azure service health in the portal

## License
This project is part of the Bajaj HackRX hackathon submission.
