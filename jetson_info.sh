#!/bin/bash

# ==============================================================================
# Jetson Information Script (Bash Version)
# ==============================================================================
# Schnelle Alternative zum Python Script - sammelt grundlegende Jetson Infos
# ==============================================================================

echo "🤖 Jetson System Information (Bash Version)"
echo "============================================================"
echo ""

# Jetson Modell erkennen
echo "📱 JETSON HARDWARE:"
echo "============================================================"
if [ -f "/proc/device-tree/model" ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null)
    echo "Modell (Device Tree): $MODEL"
else
    echo "Modell: Nicht erkannt (kein Device Tree gefunden)"
fi

# Alternative: aus cpuinfo
if grep -q "tegra" /proc/cpuinfo 2>/dev/null; then
    echo "Tegra-Prozessor: ✅ Erkannt"
else
    echo "Tegra-Prozessor: ❌ Nicht erkannt"
fi
echo ""

# JetPack Version
echo "📦 JETPACK / L4T VERSION:"
echo "============================================================"
if [ -f "/etc/nv_tegra_release" ]; then
    echo "L4T Release Info:"
    cat /etc/nv_tegra_release
else
    echo "L4T Release: Nicht gefunden"
fi
echo ""

# CUDA Version
echo "⚡ CUDA INFORMATION:"
echo "============================================================"
if command -v nvcc &> /dev/null; then
    echo "NVCC Version:"
    nvcc --version | grep "release"
else
    echo "NVCC: Nicht installiert"
fi

if [ -f "/usr/local/cuda/version.txt" ]; then
    echo "CUDA Version File:"
    cat /usr/local/cuda/version.txt
fi

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA-SMI:"
    nvidia-smi | grep "CUDA Version" | head -1
else
    echo "nvidia-smi: Nicht verfügbar"
fi
echo ""

# Python Version
echo "🐍 PYTHON INFORMATION:"
echo "============================================================"
if command -v python3 &> /dev/null; then
    echo "Python3 Version: $(python3 --version)"
    echo "Python3 Path: $(which python3)"
else
    echo "Python3: Nicht installiert"
fi

if command -v pip3 &> /dev/null; then
    echo "Pip3 Version: $(pip3 --version)"
else
    echo "Pip3: Nicht installiert"
fi
echo ""

# System Info
echo "💻 SYSTEM INFORMATION:"
echo "============================================================"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

# Speicher Info
echo "💾 SPEICHER INFORMATION:"
echo "============================================================"
if [ -f "/proc/meminfo" ]; then
    TOTAL_RAM=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
    AVAILABLE_RAM=$(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
    SWAP_TOTAL=$(grep SwapTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
    
    echo "RAM Total: $TOTAL_RAM"
    echo "RAM Verfügbar: $AVAILABLE_RAM"
    echo "Swap Total: $SWAP_TOTAL"
fi
echo ""

# Festplatte
echo "💽 FESTPLATTEN INFORMATION:"
echo "============================================================"
df -h / | head -2
echo ""

# TensorRT
echo "🚀 TENSORRT INFORMATION:"
echo "============================================================"
if dpkg -l | grep -q tensorrt 2>/dev/null; then
    echo "TensorRT Pakete:"
    dpkg -l | grep tensorrt | head -3
else
    echo "TensorRT: Nicht als Paket installiert"
fi

# Python TensorRT Check
if python3 -c "import tensorrt; print('TensorRT Python:', tensorrt.__version__)" 2>/dev/null; then
    :  # Success message already printed
else
    echo "TensorRT Python: Nicht importierbar"
fi
echo ""

# OpenCV
echo "👁️ OPENCV INFORMATION:"
echo "============================================================"
if python3 -c "import cv2; print('OpenCV Python:', cv2.__version__)" 2>/dev/null; then
    :  # Success message already printed
else
    echo "OpenCV Python: Nicht importierbar"
fi

if dpkg -l | grep -q libopencv 2>/dev/null; then
    echo "OpenCV System Pakete gefunden"
else
    echo "OpenCV System: Nicht als Paket installiert"
fi
echo ""

# PyTorch
echo "🔥 PYTORCH INFORMATION:"
echo "============================================================"
if python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    :  # Success message already printed
else
    echo "PyTorch: Nicht installiert oder nicht importierbar"
fi
echo ""

# USB Geräte (für Kameras)
echo "🔌 USB GERÄTE (Kameras):"
echo "============================================================"
if command -v lsusb &> /dev/null; then
    echo "USB Geräte (nach 'Luxonis' oder 'Camera' filternd):"
    lsusb | grep -i -E "(luxonis|camera|webcam)" || echo "Keine Kamera-USB-Geräte gefunden"
    echo ""
    echo "Alle USB Geräte:"
    lsusb | head -10
    if [ $(lsusb | wc -l) -gt 10 ]; then
        echo "... ($(lsusb | wc -l) Geräte insgesamt)"
    fi
else
    echo "lsusb: Nicht verfügbar"
fi
echo ""

# Empfehlungen basierend auf erkanntem System
echo "🎯 EMPFEHLUNGEN:"
echo "============================================================"
if grep -q -i "orin nano" /proc/device-tree/model 2>/dev/null; then
    echo "✅ Jetson Orin Nano erkannt"
    echo "   - Perfekt für YOLOv11 + TensorRT"
    echo "   - Unterstützt 3 Kameras gleichzeitig"
    echo "   - Empfohlen: PyTorch 1.13+, TensorRT 8.5+, CUDA 11.4+"
elif grep -q -i "orin" /proc/device-tree/model 2>/dev/null; then
    echo "✅ Jetson Orin Familie erkannt"
    echo "   - Beste Performance für Multi-Kamera Setup"
    echo "   - Empfohlen: Neueste Versionen aller Pakete"
elif grep -q -i "xavier" /proc/device-tree/model 2>/dev/null; then
    echo "⚡ Jetson Xavier erkannt"
    echo "   - Gut für 2-3 Kameras"
    echo "   - Eventuell niedrigere Batch-Größen verwenden"
elif grep -q -i "nano" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Jetson Nano erkannt"
    echo "   - Begrenzte Performance"
    echo "   - Maximal 1-2 Kameras empfohlen"
    echo "   - Kleinere Modelle verwenden"
else
    echo "❓ Jetson Modell nicht eindeutig erkannt"
    echo "   - Verwende Standard-Requirements"
fi
echo ""

echo "🔗 NÜTZLICHE BEFEHLE:"
echo "============================================================"
echo "Detaillierte Python-Analyse: python3 jetson_info.py"
echo "Kamera-Setup: python3 find_cameras.py"
echo "TensorRT testen: python3 print_tensorrt_version.py"
echo "System Setup: bash setup.sh"
echo ""

echo "✅ System-Scan abgeschlossen!"