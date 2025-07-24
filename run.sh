#!/bin/bash

# ==============================================================================
# System180 Demonstrator - Run Script
# ==============================================================================
# This script starts the System180 Demonstrator web application
# Just run: bash run.sh
# ==============================================================================

echo "🚀 Starting System180 Demonstrator..."
echo ""

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found! Make sure you're in the correct directory."
    exit 1
fi

# Check if required model files exist
missing_models=()
if [ ! -f "models/custom_320_FP16_detect.engine" ]; then
    missing_models+=("custom_320_FP16_detect.engine")
fi
if [ ! -f "models/NubsUpDown_320_FP16_segment.engine" ]; then
    missing_models+=("NubsUpDown_320_FP16_segment.engine")
fi
if [ ! -f "models/synthetic_320_FP16_detect.engine" ]; then
    missing_models+=("synthetic_320_FP16_detect.engine")
fi

if [ ${#missing_models[@]} -ne 0 ]; then
    echo "❌ Missing converted model files:"
    for model in "${missing_models[@]}"; do
        echo "   - models/$model"
    done
    echo ""
    echo "Please run: bash convert_models_to_engine.sh first"
    exit 1
fi

echo "✅ All model files found!"
echo ""
echo "🌐 Starting web server..."
echo "The application will be available at: http://localhost:8000"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the application"
echo "   - If port 8000 is busy, the app will try other ports"
echo "   - Make sure your cameras are connected before starting"
echo ""
echo "📊 Application is starting..."
echo "========================================="

# Start the application
python3 main.py