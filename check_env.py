#!/usr/bin/env python3
"""
Simple Environment Configuration Checker for Bajaj HackRX API
This script checks your .env file without requiring additional packages
"""

import os
import re

def read_env_file(env_file=".env"):
    """Read environment variables from .env file"""
    env_vars = {}
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        return env_vars
    except FileNotFoundError:
        print(f"❌ {env_file} file not found!")
        return {}
    except Exception as e:
        print(f"❌ Error reading {env_file}: {e}")
        return {}

def validate_configuration():
    """Validate the environment configuration"""
    print("🔍 Checking .env file configuration...\n")
    
    env_vars = read_env_file()
    if not env_vars:
        return False
    
    errors = []
    warnings = []
    
    # Required Azure OpenAI variables
    openai_vars = {
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI Service endpoint",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API key", 
        "AZURE_OPENAI_API_VERSION": "Azure OpenAI API version",
        "AZURE_OPENAI_MODEL_DEPLOYMENT": "GPT model deployment name",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "Embedding model deployment name"
    }
    
    print("📡 Azure OpenAI Configuration:")
    for var, desc in openai_vars.items():
        value = env_vars.get(var, "")
        if not value:
            errors.append(f"Missing {var} ({desc})")
        elif value.startswith("your-") or value == "":
            errors.append(f"Invalid {var}: appears to be placeholder value")
        else:
            print(f"   ✅ {var}: {value[:30]}...")
    
    # Required Azure Search variables
    search_vars = {
        "AZURE_SEARCH_ENDPOINT": "Azure Cognitive Search endpoint",
        "AZURE_SEARCH_API_KEY": "Azure Cognitive Search API key (should match your admin key)",
        "AZURE_SEARCH_INDEX_NAME": "Search index name"
    }
    
    print("\n🔍 Azure Cognitive Search Configuration:")
    for var, desc in search_vars.items():
        value = env_vars.get(var, "")
        if not value:
            errors.append(f"Missing {var} ({desc})")
        elif value.startswith("your-") or value == "":
            errors.append(f"Invalid {var}: appears to be placeholder value")
        else:
            print(f"   ✅ {var}: {value}")
    
    # Check if AZURE_SEARCH_API_KEY matches AZURE_SEARCH_ADMIN_KEY
    if env_vars.get("AZURE_SEARCH_API_KEY") != env_vars.get("AZURE_SEARCH_ADMIN_KEY"):
        warnings.append("AZURE_SEARCH_API_KEY should match AZURE_SEARCH_ADMIN_KEY for full access")
    
    # Required Azure SQL variables
    sql_vars = {
        "AZURE_SQL_SERVER": "Azure SQL Server hostname",
        "AZURE_SQL_DATABASE": "Azure SQL Database name", 
        "AZURE_SQL_USERNAME": "Azure SQL username",
        "AZURE_SQL_PASSWORD": "Azure SQL password",
        "AZURE_SQL_DRIVER": "SQL Server ODBC driver"
    }
    
    print("\n🗄️  Azure SQL Database Configuration:")
    for var, desc in sql_vars.items():
        value = env_vars.get(var, "")
        if not value:
            errors.append(f"Missing {var} ({desc})")
        elif value.startswith("your-") and var != "AZURE_SQL_DRIVER":
            errors.append(f"Invalid {var}: appears to be placeholder value - '{value}'")
        else:
            if var == "AZURE_SQL_PASSWORD":
                print(f"   ✅ {var}: {'*' * len(value)}")
            else:
                print(f"   ✅ {var}: {value}")
    
    # System configuration
    system_vars = ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K_RETRIEVAL", "SIMILARITY_THRESHOLD", "EMBEDDING_DIMENSIONS", "BEARER_TOKEN"]
    
    print("\n⚙️  System Configuration:")
    for var in system_vars:
        value = env_vars.get(var, "")
        if value:
            if var == "BEARER_TOKEN":
                print(f"   ✅ {var}: {'*' * 20}...{value[-10:]}")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            warnings.append(f"Missing {var}, will use default value")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 CONFIGURATION SUMMARY")
    print("="*60)
    
    if errors:
        print(f"❌ {len(errors)} Critical Error(s) Found:")
        for error in errors:
            print(f"   • {error}")
    
    if warnings:
        print(f"⚠️  {len(warnings)} Warning(s):")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not errors and not warnings:
        print("✅ All configuration looks good!")
    elif not errors:
        print("✅ Configuration is valid with minor warnings")
    
    print(f"\n📊 Total variables found in .env: {len(env_vars)}")
    
    # Specific issues found in your current .env
    if env_vars.get("AZURE_SQL_PASSWORD") == "your-secure-password":
        print("\n🚨 CRITICAL: Your Azure SQL password is still the placeholder!")
        print("   You need to update AZURE_SQL_PASSWORD with your actual database password")
        errors.append("Azure SQL password not updated")
    
    return len(errors) == 0

def main():
    """Main function"""
    print("Bajaj HackRX - Environment Configuration Checker")
    print("="*50)
    
    is_valid = validate_configuration()
    
    print("\n🚀 Next Steps:")
    if is_valid:
        print("1. Your configuration looks good!")
        print("2. You can now run: python bajajhackrx.py")
        print("3. Or deploy to Azure with: ./deploy-azure.bat")
    else:
        print("1. Fix the errors shown above")
        print("2. Re-run this checker: python check_env.py") 
        print("3. Deploy once all errors are resolved")
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    exit(main())
