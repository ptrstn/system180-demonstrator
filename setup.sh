#!/bin/bash

# ==============================================================================
# System180 Demonstrator - Setup Script
# ==============================================================================
# This script will install everything you need to run the System180 Demonstrator
# Just run: bash setup.sh
# ==============================================================================

echo "🚀 Setting up System180 Demonstrator..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"
echo ""

# Check if we're on the right system
echo "🔍 Checking system compatibility..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Linux system detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✅ macOS system detected"
else
    echo "⚠️  Unknown system type. This might work, but hasn't been tested."
fi
echo ""

# Install Python dependencies
echo "📦 Installing Python packages..."
echo "This might take a few minutes..."
echo ""

if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ All Python packages installed successfully!"
    else
        echo "❌ Failed to install some packages. Check the error messages above."
        exit 1
    fi
else
    echo "❌ requirements.txt file not found!"
    exit 1
fi
echo ""

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir models
    echo "✅ Models directory created"
else
    echo "✅ Models directory already exists"
fi
echo ""

echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Put your model files (.pt files) in the models/ directory"
echo "2. Run: bash convert_models.sh  (to convert models)"
echo "3. Run: bash run.sh             (to start the application)"
echo ""