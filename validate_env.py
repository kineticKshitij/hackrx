#!/usr/bin/env python3
"""
Environment Configuration Validator for Bajaj HackRX API
This script validates and loads environment variables from .env file
"""

import os
import sys
from typing import Dict, List, Optional
from dotenv import load_dotenv

class EnvironmentValidator:
    """Validates and manages environment configuration"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def load_environment(self) -> bool:
        """Load environment variables from .env file"""
        try:
            load_dotenv(self.env_file)
            print(f"✅ Loaded environment from {self.env_file}")
            return True
        except Exception as e:
            self.errors.append(f"Failed to load {self.env_file}: {str(e)}")
            return False
    
    def validate_azure_openai(self) -> bool:
        """Validate Azure OpenAI configuration"""
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": "Azure OpenAI Service endpoint",
            "AZURE_OPENAI_API_KEY": "Azure OpenAI API key",
            "AZURE_OPENAI_API_VERSION": "Azure OpenAI API version",
            "AZURE_OPENAI_MODEL_DEPLOYMENT": "GPT model deployment name",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "Embedding model deployment name"
        }
        
        valid = True
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing {var} ({description})")
                valid = False
            elif value.startswith("your-") or value == "":
                self.errors.append(f"Invalid {var}: appears to be placeholder value")
                valid = False
            else:
                print(f"✅ {var}: {value[:20]}..." if len(value) > 20 else f"✅ {var}: {value}")
        
        return valid
    
    def validate_azure_search(self) -> bool:
        """Validate Azure Cognitive Search configuration"""
        required_vars = {
            "AZURE_SEARCH_ENDPOINT": "Azure Cognitive Search endpoint",
            "AZURE_SEARCH_API_KEY": "Azure Cognitive Search API key",
            "AZURE_SEARCH_INDEX_NAME": "Search index name"
        }
        
        valid = True
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing {var} ({description})")
                valid = False
            elif value.startswith("your-") or value == "":
                self.errors.append(f"Invalid {var}: appears to be placeholder value")
                valid = False
            else:
                print(f"✅ {var}: {value}")
        
        return valid
    
    def validate_azure_sql(self) -> bool:
        """Validate Azure SQL Database configuration"""
        required_vars = {
            "AZURE_SQL_SERVER": "Azure SQL Server hostname",
            "AZURE_SQL_DATABASE": "Azure SQL Database name",
            "AZURE_SQL_USERNAME": "Azure SQL username",
            "AZURE_SQL_PASSWORD": "Azure SQL password",
            "AZURE_SQL_DRIVER": "SQL Server ODBC driver"
        }
        
        valid = True
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing {var} ({description})")
                valid = False
            elif value.startswith("your-") and var != "AZURE_SQL_DRIVER":
                self.errors.append(f"Invalid {var}: appears to be placeholder value")
                valid = False
            else:
                if var == "AZURE_SQL_PASSWORD":
                    print(f"✅ {var}: {'*' * len(value)}")
                else:
                    print(f"✅ {var}: {value}")
        
        return valid
    
    def validate_system_config(self) -> bool:
        """Validate system configuration"""
        config_vars = {
            "CHUNK_SIZE": (int, 256, 2048),
            "CHUNK_OVERLAP": (int, 0, 200),
            "TOP_K_RETRIEVAL": (int, 1, 50),
            "SIMILARITY_THRESHOLD": (float, 0.0, 1.0),
            "EMBEDDING_DIMENSIONS": (int, 1000, 2000),
            "BEARER_TOKEN": (str, None, None)
        }
        
        valid = True
        for var, (var_type, min_val, max_val) in config_vars.items():
            value = os.getenv(var)
            if not value:
                self.warnings.append(f"Missing {var}, will use default")
                continue
            
            try:
                if var_type == int:
                    typed_value = int(value)
                    if min_val and typed_value < min_val:
                        self.warnings.append(f"{var} ({typed_value}) is below recommended minimum ({min_val})")
                    elif max_val and typed_value > max_val:
                        self.warnings.append(f"{var} ({typed_value}) is above recommended maximum ({max_val})")
                elif var_type == float:
                    typed_value = float(value)
                    if min_val and typed_value < min_val:
                        self.warnings.append(f"{var} ({typed_value}) is below recommended minimum ({min_val})")
                    elif max_val and typed_value > max_val:
                        self.warnings.append(f"{var} ({typed_value}) is above recommended maximum ({max_val})")
                
                if var == "BEARER_TOKEN":
                    print(f"✅ {var}: {'*' * 20}...{value[-10:]}")
                else:
                    print(f"✅ {var}: {value}")
                    
            except ValueError:
                self.errors.append(f"Invalid {var}: expected {var_type.__name__}, got '{value}'")
                valid = False
        
        return valid
    
    def validate_all(self) -> bool:
        """Validate all configuration sections"""
        print("🔍 Validating environment configuration...\n")
        
        if not self.load_environment():
            return False
        
        print("\n📡 Azure OpenAI Configuration:")
        openai_valid = self.validate_azure_openai()
        
        print("\n🔍 Azure Cognitive Search Configuration:")
        search_valid = self.validate_azure_search()
        
        print("\n🗄️  Azure SQL Database Configuration:")
        sql_valid = self.validate_azure_sql()
        
        print("\n⚙️  System Configuration:")
        system_valid = self.validate_system_config()
        
        return openai_valid and search_valid and sql_valid and system_valid
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("🎯 ENVIRONMENT VALIDATION SUMMARY")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("✅ All configuration is valid! Ready for deployment.")
        else:
            if self.errors:
                print(f"❌ {len(self.errors)} Error(s) found:")
                for error in self.errors:
                    print(f"   • {error}")
            
            if self.warnings:
                print(f"⚠️  {len(self.warnings)} Warning(s):")
                for warning in self.warnings:
                    print(f"   • {warning}")
        
        print("\n🚀 Next Steps:")
        if self.errors:
            print("1. Fix the errors listed above")
            print("2. Re-run this validation script")
            print("3. Deploy to Azure once validation passes")
        else:
            print("1. Run: python bajajhackrx.py (for local testing)")
            print("2. Or run: ./deploy-azure.bat (for Azure deployment)")
            print("3. Test API at: http://localhost:8000/docs")

def main():
    """Main validation function"""
    validator = EnvironmentValidator()
    is_valid = validator.validate_all()
    validator.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()
