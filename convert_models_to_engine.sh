#!/bin/bash

# ==============================================================================
# System180 Demonstrator - Model Conversion Script
# ==============================================================================
# This script converts your YOLO models (.pt files) to TensorRT engines (.engine files)
# Make sure you have the following files in the models/ directory:
# - custom.pt
# - NubsUpDown.pt  
# - synthetic.pt
# ==============================================================================

echo "🔧 Converting YOLO models to TensorRT engines..."
echo "This will take several minutes per model..."
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "❌ Models directory not found! Run setup.sh first."
    exit 1
fi

# Check if required model files exist
missing_files=()
if [ ! -f "models/custom.pt" ]; then
    missing_files+=("custom.pt")
fi
if [ ! -f "models/NubsUpDown.pt" ]; then
    missing_files+=("NubsUpDown.pt")
fi
if [ ! -f "models/synthetic.pt" ]; then
    missing_files+=("synthetic.pt")
fi

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing model files in models/ directory:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please add these files to the models/ directory and run this script again."
    exit 1
fi

echo "✅ All required model files found!"
echo ""

# Convert custom model
echo "🔄 Converting custom.pt (1/3)..."
ultralytics export model=./models/custom.pt format=engine device=0 imgsz=320 half=True
if [ $? -eq 0 ]; then
    mv ./models/custom.engine ./models/custom_320_FP16_detect.engine
    echo "✅ custom.pt converted successfully!"
else
    echo "❌ Failed to convert custom.pt"
    exit 1
fi
echo ""

# Convert NubsUpDown model
echo "🔄 Converting NubsUpDown.pt (2/3)..."
ultralytics export model=./models/NubsUpDown.pt format=engine device=0 imgsz=320 half=True
if [ $? -eq 0 ]; then
    mv ./models/NubsUpDown.engine ./models/NubsUpDown_320_FP16_segment.engine
    echo "✅ NubsUpDown.pt converted successfully!"
else
    echo "❌ Failed to convert NubsUpDown.pt"
    exit 1
fi
echo ""

# Convert synthetic model
echo "🔄 Converting synthetic.pt (3/3)..."
ultralytics export model=./models/synthetic.pt format=engine device=0 imgsz=320 half=True
if [ $? -eq 0 ]; then
    mv ./models/synthetic.engine ./models/synthetic_320_FP16_detect.engine
    echo "✅ synthetic.pt converted successfully!"
else
    echo "❌ Failed to convert synthetic.pt"
    exit 1
fi
echo ""

echo "🎉 All models converted successfully!"
echo ""
echo "Generated files:"
echo "✅ custom_320_FP16_detect.engine"
echo "✅ NubsUpDown_320_FP16_segment.engine" 
echo "✅ synthetic_320_FP16_detect.engine"
echo ""
echo "You can now run: bash run.sh"