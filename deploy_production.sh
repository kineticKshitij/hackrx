#!/usr/bin/env bash
# Bajaj HackRX - Full Azure Deployment Script
# This script sets up the complete Azure environment for the full application

echo "=========================================="
echo "Bajaj HackRX - Azure Deployment Setup"
echo "=========================================="

# Configuration
RESOURCE_GROUP="bajaj-hackrx-production"
LOCATION="centralindia"
APP_NAME="bajaj-hackrx-production"
PLAN_NAME="bajaj-hackrx-plan-premium"

echo "Setting up Azure resources for production deployment..."

# 1. Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Create App Service Plan (Standard tier for production)
echo "Creating App Service Plan..."
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku S1 \
    --is-linux

# 3. Create Web App
echo "Creating Web App..."
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --runtime "PYTHON:3.11"

# 4. Configure application settings
echo "Configuring application settings..."
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings \
        SCM_DO_BUILD_DURING_DEPLOYMENT=true \
        ENABLE_ORYX_BUILD=true \
        POST_BUILD_SCRIPT_PATH="post_build.sh"

# 5. Set startup command
echo "Setting startup command..."
az webapp config set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --startup-file "python -m uvicorn bajajhackrx:app --host 0.0.0.0 --port 8000"

# 6. Enable logging
echo "Enabling application logging..."
az webapp log config \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --application-logging filesystem \
    --level information

echo "=========================================="
echo "Next Steps for Full Production Setup:"
echo "=========================================="
echo ""
echo "1. Set up Azure OpenAI Service:"
echo "   az cognitiveservices account create --name 'bajaj-openai' --resource-group $RESOURCE_GROUP --kind OpenAI --sku S0 --location eastus"
echo ""
echo "2. Set up Azure Cognitive Search:"
echo "   az search service create --name 'bajaj-search' --resource-group $RESOURCE_GROUP --sku Standard --location $LOCATION"
echo ""
echo "3. Set up Azure SQL Database:"
echo "   az sql server create --name 'bajaj-sql-server' --resource-group $RESOURCE_GROUP --location $LOCATION --admin-user 'sqladmin' --admin-password 'YourPassword123!'"
echo "   az sql db create --name 'bajaj-hackrx-db' --resource-group $RESOURCE_GROUP --server 'bajaj-sql-server' --service-objective S0"
echo ""
echo "4. Configure environment variables:"
echo "   az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $APP_NAME --settings \\"
echo "     AZURE_OPENAI_ENDPOINT='https://bajaj-openai.openai.azure.com/' \\"
echo "     AZURE_OPENAI_API_KEY='your-openai-key' \\"
echo "     AZURE_SEARCH_ENDPOINT='https://bajaj-search.search.windows.net' \\"
echo "     AZURE_SEARCH_API_KEY='your-search-key' \\"
echo "     AZURE_SQL_SERVER='bajaj-sql-server.database.windows.net' \\"
echo "     AZURE_SQL_DATABASE='bajaj-hackrx-db' \\"
echo "     AZURE_SQL_USERNAME='sqladmin' \\"
echo "     AZURE_SQL_PASSWORD='YourPassword123!'"
echo ""
echo "5. Deploy the application:"
echo "   az webapp up --name $APP_NAME --resource-group $RESOURCE_GROUP --runtime 'PYTHON:3.11'"
echo ""
echo "=========================================="
echo "Production app will be available at:"
echo "https://$APP_NAME.azurewebsites.net"
echo "=========================================="
