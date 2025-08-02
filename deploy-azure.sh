#!/bin/bash

# Azure Deployment Script for Bajaj HackRX API
# This script deploys the application to Azure Container Apps

set -e

# Configuration
RESOURCE_GROUP="bajaj-hackrx-rg"
LOCATION="eastus"
CONTAINER_APP_ENV="bajaj-hackrx-env"
CONTAINER_APP_NAME="bajaj-hackrx-api"
ACR_NAME="bajajhackrxacr"

echo "🚀 Starting Azure deployment for Bajaj HackRX API..."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it first."
    exit 1
fi

# Login to Azure (if not already logged in)
echo "🔐 Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "Please login to Azure:"
    az login
fi

# Create resource group
echo "📦 Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo "🏗️  Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
echo "📝 ACR Login Server: $ACR_LOGIN_SERVER"

# Build and push Docker image
echo "🔨 Building and pushing Docker image..."
az acr build --registry $ACR_NAME --image bajaj-hackrx-api:latest .

# Create Container App Environment
echo "🌍 Creating Container App Environment..."
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# Create Container App
echo "🚢 Creating Container App..."
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/bajaj-hackrx-api:latest \
    --target-port 8000 \
    --ingress 'external' \
    --registry-server $ACR_LOGIN_SERVER \
    --min-replicas 1 \
    --max-replicas 10 \
    --cpu 1.0 \
    --memory 2.0Gi

# Get the application URL
APP_URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" --output tsv)

echo "✅ Deployment completed successfully!"
echo "🌐 Application URL: https://$APP_URL"
echo "📖 API Documentation: https://$APP_URL/docs"
echo "❤️  Health Check: https://$APP_URL/health"
echo ""
echo "⚠️  Next Steps:"
echo "1. Configure your Azure services (OpenAI, Cognitive Search, SQL Database)"
echo "2. Update environment variables in the Container App"
echo "3. Test the API endpoints"
echo ""
echo "🔧 To update environment variables, run:"
echo "az containerapp update --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --set-env-vars KEY=VALUE"
