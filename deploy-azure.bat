@echo off
REM Azure Deployment Script for Bajaj HackRX API (Windows)
REM This script deploys the application to Azure Container Apps

echo 🚀 Starting Azure deployment for Bajaj HackRX API...

REM Configuration
set RESOURCE_GROUP=bajaj-hackrx-rg
set LOCATION=eastus
set CONTAINER_APP_ENV=bajaj-hackrx-env
set CONTAINER_APP_NAME=bajaj-hackrx-api
set ACR_NAME=bajajhackrxacr

REM Check if Azure CLI is installed
az --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Azure CLI is not installed. Please install it first.
    exit /b 1
)

REM Check Azure login status
echo 🔐 Checking Azure login status...
az account show >nul 2>&1
if %errorlevel% neq 0 (
    echo Please login to Azure:
    az login
)

REM Create resource group
echo 📦 Creating resource group...
az group create --name %RESOURCE_GROUP% --location %LOCATION%

REM Create Azure Container Registry
echo 🏗️  Creating Azure Container Registry...
az acr create --resource-group %RESOURCE_GROUP% --name %ACR_NAME% --sku Basic --admin-enabled true

REM Get ACR login server
for /f "tokens=*" %%i in ('az acr show --name %ACR_NAME% --resource-group %RESOURCE_GROUP% --query "loginServer" --output tsv') do set ACR_LOGIN_SERVER=%%i
echo 📝 ACR Login Server: %ACR_LOGIN_SERVER%

REM Build and push Docker image
echo 🔨 Building and pushing Docker image...
az acr build --registry %ACR_NAME% --image bajaj-hackrx-api:latest .

REM Create Container App Environment
echo 🌍 Creating Container App Environment...
az containerapp env create --name %CONTAINER_APP_ENV% --resource-group %RESOURCE_GROUP% --location %LOCATION%

REM Create Container App
echo 🚢 Creating Container App...
az containerapp create --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --environment %CONTAINER_APP_ENV% --image %ACR_LOGIN_SERVER%/bajaj-hackrx-api:latest --target-port 8000 --ingress external --registry-server %ACR_LOGIN_SERVER% --min-replicas 1 --max-replicas 10 --cpu 1.0 --memory 2.0Gi

REM Get the application URL
for /f "tokens=*" %%i in ('az containerapp show --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --query "properties.configuration.ingress.fqdn" --output tsv') do set APP_URL=%%i

echo ✅ Deployment completed successfully!
echo 🌐 Application URL: https://%APP_URL%
echo 📖 API Documentation: https://%APP_URL%/docs
echo ❤️  Health Check: https://%APP_URL%/health
echo.
echo ⚠️  Next Steps:
echo 1. Configure your Azure services (OpenAI, Cognitive Search, SQL Database)
echo 2. Update environment variables in the Container App
echo 3. Test the API endpoints
echo.
echo 🔧 To update environment variables, run:
echo az containerapp update --name %CONTAINER_APP_NAME% --resource-group %RESOURCE_GROUP% --set-env-vars KEY=VALUE

pause
