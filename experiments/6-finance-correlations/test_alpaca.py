#!/usr/bin/env python3
"""
Simple test to verify Alpaca credentials are loaded from .env
"""

import os
from dotenv import load_dotenv

def test_alpaca_credentials():
    print("🧪 Testing Alpaca Credentials Loading...")
    
    # Load .env file
    load_dotenv()
    
    # Check if credentials are loaded
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    print(f"  📝 API Key found: {'Yes' if api_key else 'No'}")
    print(f"  📝 Secret Key found: {'Yes' if secret_key else 'No'}")
    
    if api_key and secret_key:
        print(f"  ✅ API Key: {api_key[:10]}...")
        print(f"  ✅ Secret Key: {secret_key[:10]}...")
        return True
    else:
        print("  ❌ Missing Alpaca credentials")
        return False

if __name__ == "__main__":
    test_alpaca_credentials() 