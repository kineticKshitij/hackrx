@echo off
REM Azure App Service Deployment Script for Bajaj HackRX API

echo 🚀 Starting Azure App Service deployment for Bajaj HackRX API...

REM Configuration
set RESOURCE_GROUP=bajaj-hackrx-rg
set LOCATION=centralindia
set APP_SERVICE_PLAN=bajaj-hackrx-plan
set WEB_APP_NAME=bajaj-hackrx-api-%RANDOM%

echo 📦 Creating App Service Plan...
az appservice plan create ^
    --name %APP_SERVICE_PLAN% ^
    --resource-group %RESOURCE_GROUP% ^
    --location %LOCATION% ^
    --sku FREE ^
    --is-linux

if %errorlevel% neq 0 (
    echo ❌ App Service Plan creation failed!
    pause
    exit /b 1
)

echo 🌐 Creating Web App...
az webapp create ^
    --resource-group %RESOURCE_GROUP% ^
    --plan %APP_SERVICE_PLAN% ^
    --name %WEB_APP_NAME% ^
    --runtime "PYTHON:3.11" ^
    --startup-file "python app_simple.py"

if %errorlevel% neq 0 (
    echo ❌ Web App creation failed!
    pause
    exit /b 1
)

echo 🔧 Configuring environment variables...
az webapp config appsettings set ^
    --resource-group %RESOURCE_GROUP% ^
    --name %WEB_APP_NAME% ^
    --settings ^
        PORT=8000 ^
        AZURE_OPENAI_ENDPOINT="%AZURE_OPENAI_ENDPOINT%" ^
        AZURE_OPENAI_API_KEY="%AZURE_OPENAI_API_KEY%"

echo 📤 Deploying application code...
az webapp up ^
    --resource-group %RESOURCE_GROUP% ^
    --name %WEB_APP_NAME% ^
    --plan %APP_SERVICE_PLAN% ^
    --location %LOCATION% ^
    --runtime "PYTHON:3.11"

if %errorlevel% neq 0 (
    echo ❌ Deployment failed!
    pause
    exit /b 1
)

echo ✅ Deployment completed successfully!
echo 🌐 Application URL: https://%WEB_APP_NAME%.azurewebsites.net
echo 📖 API Documentation: https://%WEB_APP_NAME%.azurewebsites.net/docs
echo ❤️  Health Check: https://%WEB_APP_NAME%.azurewebsites.net/health
echo.
echo 🎉 Your Bajaj HackRX API is now running on Azure App Service!
echo.
echo 🔍 To view logs, run:
echo az webapp log tail --name %WEB_APP_NAME% --resource-group %RESOURCE_GROUP%
echo.
pause
