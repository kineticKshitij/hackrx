#!/bin/bash

echo "🚀 Bajaj HackRX - Secure Setup Script"
echo "====================================="
echo

# Check if .env already exists
if [[ -f .env ]]; then
    read -p "⚠️  .env file already exists! Do you want to overwrite it? (y/N): " overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 1
    fi
fi

# Copy example to .env
cp .env.example .env
echo "✅ Created .env file from template"

echo
echo "🔧 IMPORTANT: You must now edit the .env file with your actual Azure credentials!"
echo
echo "Required updates:"
echo "  1. AZURE_OPENAI_ENDPOINT - Your Azure OpenAI service endpoint"
echo "  2. AZURE_OPENAI_API_KEY - Your Azure OpenAI API key"
echo "  3. AZURE_SEARCH_ENDPOINT - Your Azure Cognitive Search endpoint"
echo "  4. AZURE_SEARCH_API_KEY - Your Azure Search admin key"
echo "  5. AZURE_SQL_SERVER - Your Azure SQL server hostname"
echo "  6. AZURE_SQL_DATABASE - Your Azure SQL database name"
echo "  7. AZURE_SQL_USERNAME - Your Azure SQL username"
echo "  8. AZURE_SQL_PASSWORD - Your Azure SQL password"
echo

# Try to open with different editors
if command -v code &> /dev/null; then
    echo "📝 Opening .env file in VS Code..."
    code .env
elif command -v nano &> /dev/null; then
    echo "📝 Opening .env file in nano..."
    nano .env
elif command -v vim &> /dev/null; then
    echo "📝 Opening .env file in vim..."
    vim .env
else
    echo "📝 Please manually edit the .env file with your preferred editor"
fi

echo
echo "🔍 Validating configuration..."
python3 check_env.py || python check_env.py

if [[ $? -eq 0 ]]; then
    echo
    echo "✅ Setup completed successfully!"
    echo "🚀 You can now run: python bajajhackrx.py"
    echo "📖 Or view API docs at: http://localhost:8000/docs"
else
    echo
    echo "❌ Configuration validation failed!"
    echo "Please fix the issues above and run: python check_env.py"
fi

echo
