@echo off
REM Simple Azure Deployment Script for Bajaj HackRX API (Windows)
REM This script deploys the application to Azure Container Instances

echo 🚀 Starting simplified Azure deployment for Bajaj HackRX API...

REM Configuration
set RESOURCE_GROUP=bajaj-hackrx-rg
set LOCATION=centralindia
set CONTAINER_NAME=bajaj-hackrx-api
set IMAGE_NAME=kshitij2025/bajaj-hackrx-api:latest

echo 📦 Building Docker image locally...
docker build -t %IMAGE_NAME% .

if %errorlevel% neq 0 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo 📤 Pushing image to Docker Hub...
docker push %IMAGE_NAME%

if %errorlevel% neq 0 (
    echo ❌ Docker push failed! Make sure you're logged in to Docker Hub
    echo Run: docker login
    pause
    exit /b 1
)

echo 🚢 Creating Azure Container Instance...
az container create ^
    --resource-group %RESOURCE_GROUP% ^
    --name %CONTAINER_NAME% ^
    --image %IMAGE_NAME% ^
    --location %LOCATION% ^
    --ports 8000 ^
    --protocol TCP ^
    --ip-address Public ^
    --cpu 2 ^
    --memory 4 ^
    --environment-variables ^
        AZURE_OPENAI_ENDPOINT=%AZURE_OPENAI_ENDPOINT% ^
        AZURE_OPENAI_API_KEY=%AZURE_OPENAI_API_KEY% ^
        AZURE_OPENAI_API_VERSION=%AZURE_OPENAI_API_VERSION% ^
        AZURE_OPENAI_MODEL_DEPLOYMENT=%AZURE_OPENAI_MODEL_DEPLOYMENT% ^
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT=%AZURE_OPENAI_EMBEDDING_DEPLOYMENT% ^
        AZURE_SEARCH_ENDPOINT=%AZURE_SEARCH_ENDPOINT% ^
        AZURE_SEARCH_API_KEY=%AZURE_SEARCH_API_KEY% ^
        AZURE_SEARCH_INDEX_NAME=%AZURE_SEARCH_INDEX_NAME% ^
        AZURE_SQL_SERVER=%AZURE_SQL_SERVER% ^
        AZURE_SQL_DATABASE=%AZURE_SQL_DATABASE% ^
        AZURE_SQL_USERNAME=%AZURE_SQL_USERNAME% ^
        AZURE_SQL_PASSWORD=%AZURE_SQL_PASSWORD% ^
        AZURE_SQL_DRIVER="ODBC Driver 18 for SQL Server"

if %errorlevel% neq 0 (
    echo ❌ Container deployment failed!
    pause
    exit /b 1
)

REM Get the container IP
for /f "tokens=*" %%i in ('az container show --resource-group %RESOURCE_GROUP% --name %CONTAINER_NAME% --query "ipAddress.ip" --output tsv') do set CONTAINER_IP=%%i

echo ✅ Deployment completed successfully!
echo 🌐 Application URL: http://%CONTAINER_IP%:8000
echo 📖 API Documentation: http://%CONTAINER_IP%:8000/docs
echo ❤️  Health Check: http://%CONTAINER_IP%:8000/health
echo.
echo 🎉 Your Bajaj HackRX API is now running on Azure!
echo.
pause
