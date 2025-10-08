#!/bin/bash

# Face Recognition System - Environment Setup Script
# This script helps set up the environment configuration

set -e  # Exit on any error

echo "🔧 Face Recognition System - Environment Setup"
echo "=============================================="

# Check if .env exists
if [ -f ".env" ]; then
    echo "✅ .env file already exists"
    read -p "Do you want to overwrite it? (y/N): " overwrite
    if [[ $overwrite != "y" && $overwrite != "Y" ]]; then
        echo "Setup cancelled. Existing .env file preserved."
        exit 0
    fi
fi

# Copy example file
if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
else
    echo "❌ .env.example not found!"
    exit 1
fi

# Set proper permissions
chmod 600 .env
echo "✅ Set secure permissions on .env file"

echo ""
echo "🔐 IMPORTANT: Please update the following in your .env file:"
echo "=============================================="
echo "1. AZURE_STORAGE_CONNECTION_STRING - Your Azure storage connection string"
echo "2. API_SECRET_KEY - Change from default value"
echo "3. SMTP credentials - If you want email notifications"
echo ""

# Prompt for Azure connection string
echo "📝 Quick Setup (Optional):"
echo "========================="
read -p "Enter your Azure Storage connection string (or press Enter to skip): " azure_conn

if [ ! -z "$azure_conn" ]; then
    # Update the connection string in .env
    if command -v sed &> /dev/null; then
        sed -i "s|AZURE_STORAGE_CONNECTION_STRING=.*|AZURE_STORAGE_CONNECTION_STRING=$azure_conn|" .env
        echo "✅ Updated Azure connection string"
    else
        echo "⚠️  Please manually update AZURE_STORAGE_CONNECTION_STRING in .env"
    fi
fi

# Generate a random API key
if command -v openssl &> /dev/null; then
    api_key=$(openssl rand -hex 32)
    sed -i "s|API_SECRET_KEY=.*|API_SECRET_KEY=$api_key|" .env
    echo "✅ Generated random API secret key"
elif command -v python3 &> /dev/null; then
    api_key=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s|API_SECRET_KEY=.*|API_SECRET_KEY=$api_key|" .env
    echo "✅ Generated random API secret key"
else
    echo "⚠️  Please manually update API_SECRET_KEY in .env"
fi

echo ""
echo "🧪 Testing Configuration:"
echo "========================"

# Test if python and required packages are available
if [ -f "env/bin/python" ]; then
    echo "Testing configuration..."
    if env/bin/python config.py; then
        echo "✅ Configuration test passed!"
    else
        echo "❌ Configuration test failed. Please check your .env file."
        exit 1
    fi
else
    echo "⚠️  Virtual environment not found. Please run:"
    echo "   python -m venv env"
    echo "   source env/bin/activate"
    echo "   pip install -r requirements.txt"
fi

echo ""
echo "🎉 Environment setup complete!"
echo "=============================="
echo "Next steps:"
echo "1. Review and update .env file with your specific values"
echo "2. Test the system: python main_auth.py --mode test-azure"
echo "3. For production: Set proper system environment variables"
echo ""
echo "📚 For more information, see:"
echo "   - SECURITY.md for security best practices"
echo "   - README.md for usage instructions"
echo ""
echo "🔒 Security reminder: Never commit .env to version control!"