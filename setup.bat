@echo off
echo 🚀 Bajaj HackRX - Secure Setup Script
echo =====================================
echo.

REM Check if .env already exists
if exist .env (
    echo ⚠️  .env file already exists!
    set /p overwrite="Do you want to overwrite it? (y/N): "
    if /i not "%overwrite%"=="y" (
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

REM Copy example to .env
copy .env.example .env >nul
echo ✅ Created .env file from template

echo.
echo 🔧 IMPORTANT: You must now edit the .env file with your actual Azure credentials!
echo.
echo Required updates:
echo   1. AZURE_OPENAI_ENDPOINT - Your Azure OpenAI service endpoint
echo   2. AZURE_OPENAI_API_KEY - Your Azure OpenAI API key
echo   3. AZURE_SEARCH_ENDPOINT - Your Azure Cognitive Search endpoint  
echo   4. AZURE_SEARCH_API_KEY - Your Azure Search admin key
echo   5. AZURE_SQL_SERVER - Your Azure SQL server hostname
echo   6. AZURE_SQL_DATABASE - Your Azure SQL database name
echo   7. AZURE_SQL_USERNAME - Your Azure SQL username
echo   8. AZURE_SQL_PASSWORD - Your Azure SQL password
echo.
echo 📝 Opening .env file for editing...
notepad .env

echo.
echo 🔍 Validating configuration...
python check_env.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Setup completed successfully!
    echo 🚀 You can now run: python bajajhackrx.py
    echo 📖 Or view API docs at: http://localhost:8000/docs
) else (
    echo.
    echo ❌ Configuration validation failed!
    echo Please fix the issues above and run: python check_env.py
)

echo.
pause
