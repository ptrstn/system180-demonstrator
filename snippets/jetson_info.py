#!/usr/bin/env python3

"""
Jetson System Information Script
===============================
Zeigt alle wichtigen Informationen über dein Jetson System an,
damit du die richtigen Requirements installieren kannst.
"""

import os
import subprocess
import platform
import sys
import re
from pathlib import Path

def run_command(cmd, capture_output=True, text=True):
    """Führt einen Befehl aus und gibt das Ergebnis zurück"""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=text)
        else:
            result = subprocess.run(cmd, capture_output=capture_output, text=text)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def get_jetson_model():
    """Bestimmt das Jetson Modell"""
    # Versuche verschiedene Methoden
    methods = [
        ("cat /proc/device-tree/model", "Device Tree"),
        ("cat /proc/cpuinfo | grep 'Model'", "CPU Info"),
        ("sudo dmidecode -s system-product-name 2>/dev/null", "DMI"),
    ]
    
    for cmd, method in methods:
        result = run_command(cmd)
        if result:
            # Bekannte Jetson Modelle erkennen
            result_lower = result.lower()
            if 'orin nano' in result_lower:
                return "Jetson Orin Nano", method
            elif 'orin nx' in result_lower:
                return "Jetson Orin NX", method
            elif 'orin agx' in result_lower or 'agx orin' in result_lower:
                return "Jetson AGX Orin", method
            elif 'xavier nx' in result_lower:
                return "Jetson Xavier NX", method
            elif 'xavier agx' in result_lower or 'agx xavier' in result_lower:
                return "Jetson AGX Xavier", method
            elif 'nano' in result_lower:
                return "Jetson Nano", method
            elif 'tx2' in result_lower:
                return "Jetson TX2", method
            elif 'tx1' in result_lower:
                return "Jetson TX1", method
            
            return f"Unbekanntes Jetson ({result})", method
    
    return "Nicht erkannt", "N/A"

def get_jetpack_version():
    """Bestimmt die JetPack Version"""
    # JetPack Version aus verschiedenen Quellen
    jetpack_files = [
        "/etc/nv_tegra_release",
        "/etc/nv_boot_control.conf"
    ]
    
    # Methode 1: nv_tegra_release
    if os.path.exists("/etc/nv_tegra_release"):
        content = run_command("cat /etc/nv_tegra_release")
        if content:
            # Beispiel: "# R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t194ref, EABI: aarch64, DATE: Tue Aug  1 19:57:35 UTC 2023"
            match = re.search(r'R(\d+).*?REVISION:\s*([\d.]+)', content)
            if match:
                r_version = match.group(1)
                revision = match.group(2)
                return f"JetPack 5.x (R{r_version}.{revision})", content
    
    # Methode 2: dpkg
    jetpack_pkg = run_command("dpkg -l | grep nvidia-jetpack")
    if jetpack_pkg:
        return f"JetPack (dpkg): {jetpack_pkg}", jetpack_pkg
    
    # Methode 3: L4T Release
    l4t_release = run_command("cat /etc/nv_tegra_release 2>/dev/null || echo 'Nicht gefunden'")
    
    return "Nicht erkannt", l4t_release

def get_cuda_version():
    """Bestimmt CUDA Version"""
    methods = [
        ("nvcc --version 2>/dev/null", "NVCC"),
        ("cat /usr/local/cuda/version.txt 2>/dev/null", "Version File"),
        ("nvidia-smi 2>/dev/null | grep 'CUDA Version'", "nvidia-smi")
    ]
    
    versions = {}
    for cmd, method in methods:
        result = run_command(cmd)
        if result:
            # CUDA Version extrahieren
            cuda_match = re.search(r'(\d+\.\d+)', result)
            if cuda_match:
                versions[method] = cuda_match.group(1)
    
    return versions

def get_tensorrt_version():
    """Bestimmt TensorRT Version"""
    methods = [
        ("dpkg -l | grep tensorrt", "Package Manager"),
        ("python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null", "Python"),
    ]
    
    versions = {}
    for cmd, method in methods:
        result = run_command(cmd)
        if result:
            if method == "Python":
                versions[method] = result
            else:
                # Aus Package Namen extrahieren
                trt_match = re.search(r'tensorrt.*?(\d+\.\d+\.\d+)', result)
                if trt_match:
                    versions[method] = trt_match.group(1)
    
    return versions

def get_opencv_version():
    """Bestimmt OpenCV Version"""
    # Python OpenCV
    opencv_py = run_command("python3 -c 'import cv2; print(cv2.__version__)' 2>/dev/null")
    
    # System OpenCV
    opencv_pkg = run_command("dpkg -l | grep libopencv")
    
    return {
        "Python": opencv_py if opencv_py else "Nicht installiert",
        "System": opencv_pkg if opencv_pkg else "Nicht gefunden"
    }

def get_python_version():
    """Python Version Info"""
    return {
        "Version": platform.python_version(),
        "Executable": sys.executable,
        "Architecture": platform.architecture()[0]
    }

def get_system_info():
    """Grundlegende System-Informationen"""
    return {
        "OS": platform.system(),
        "OS Version": platform.release(),
        "Architecture": platform.machine(),
        "Hostname": platform.node(),
    }

def get_memory_info():
    """RAM und Swap Information"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = re.search(r'MemTotal:\s+(\d+)', meminfo)
        mem_available = re.search(r'MemAvailable:\s+(\d+)', meminfo)
        swap_total = re.search(r'SwapTotal:\s+(\d+)', meminfo)
        
        return {
            "RAM Total": f"{int(mem_total.group(1))/1024/1024:.1f} GB" if mem_total else "Unknown",
            "RAM Available": f"{int(mem_available.group(1))/1024/1024:.1f} GB" if mem_available else "Unknown",
            "Swap Total": f"{int(swap_total.group(1))/1024/1024:.1f} GB" if swap_total else "Unknown"
        }
    except:
        return {"Error": "Could not read memory info"}

def get_storage_info():
    """Speicher-Information"""
    disk_usage = run_command("df -h / | tail -1")
    if disk_usage:
        parts = disk_usage.split()
        return {
            "Root Filesystem": parts[0] if len(parts) > 0 else "Unknown",
            "Total Size": parts[1] if len(parts) > 1 else "Unknown",
            "Used": parts[2] if len(parts) > 2 else "Unknown",
            "Available": parts[3] if len(parts) > 3 else "Unknown",
            "Usage": parts[4] if len(parts) > 4 else "Unknown"
        }
    return {"Error": "Could not read storage info"}

def get_requirements_recommendation(jetson_model, jetpack_version):
    """Empfehlung für Requirements basierend auf Hardware"""
    recommendations = {
        "Jetson Orin Nano": {
            "Python": ">=3.8",
            "PyTorch": "torch>=1.13.0 (with CUDA support)",
            "TensorRT": ">=8.5.0",
            "OpenCV": ">=4.5.0 (with CUDA support)",
            "CUDA": ">=11.4",
            "Notes": "Perfekt für YOLOv11 und TensorRT. Genug Power für 3 Kameras."
        },
        "Jetson AGX Orin": {
            "Python": ">=3.8",
            "PyTorch": "torch>=1.13.0 (with CUDA support)",
            "TensorRT": ">=8.5.0",
            "OpenCV": ">=4.5.0 (with CUDA support)",
            "CUDA": ">=11.4",
            "Notes": "Beste Performance. Kann mehrere Modelle parallel verarbeiten."
        },
        "Jetson Xavier NX": {
            "Python": ">=3.6",
            "PyTorch": "torch>=1.10.0 (with CUDA support)",
            "TensorRT": ">=8.0.0",
            "OpenCV": ">=4.2.0 (with CUDA support)",
            "CUDA": ">=10.2",
            "Notes": "Gut für kleinere Modelle. Eventuell nur 2 Kameras gleichzeitig."
        },
        "Jetson Nano": {
            "Python": ">=3.6",
            "PyTorch": "torch>=1.8.0",
            "TensorRT": ">=7.1.0",
            "OpenCV": ">=4.1.0",
            "CUDA": ">=10.2",
            "Notes": "Begrenzte Performance. Nur 1-2 Kameras empfohlen."
        }
    }
    
    return recommendations.get(jetson_model, {
        "Notes": "Unbekanntes Jetson Modell. Verwende Standard-Requirements."
    })

def print_section(title, data):
    """Formatierte Ausgabe einer Sektion"""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print('='*60)
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key:20}: {value}")
    elif isinstance(data, tuple):
        print(f"Wert: {data[0]}")
        print(f"Methode: {data[1]}")
    else:
        print(data)

def main():
    print("🤖 Jetson System Information Scanner")
    print("=" * 60)
    print("Sammle Informationen über dein Jetson System...")
    print("Dies kann einen Moment dauern...")
    
    # Hardware Info
    jetson_model, detection_method = get_jetson_model()
    print_section("JETSON HARDWARE", {
        "Modell": jetson_model,
        "Erkannt via": detection_method
    })
    
    # Software Versionen
    jetpack_version, jp_details = get_jetpack_version()
    print_section("JETPACK VERSION", {
        "Version": jetpack_version,
        "Details": jp_details[:100] + "..." if len(jp_details) > 100 else jp_details
    })
    
    # CUDA
    cuda_versions = get_cuda_version()
    print_section("CUDA VERSIONEN", cuda_versions if cuda_versions else {"Status": "Nicht installiert"})
    
    # TensorRT
    trt_versions = get_tensorrt_version()
    print_section("TENSORRT VERSIONEN", trt_versions if trt_versions else {"Status": "Nicht installiert"})
    
    # OpenCV
    opencv_versions = get_opencv_version()
    print_section("OPENCV VERSIONEN", opencv_versions)
    
    # Python
    python_info = get_python_version()
    print_section("PYTHON INFORMATION", python_info)
    
    # System
    system_info = get_system_info()
    print_section("SYSTEM INFORMATION", system_info)
    
    # Memory
    memory_info = get_memory_info()
    print_section("SPEICHER INFORMATION", memory_info)
    
    # Storage
    storage_info = get_storage_info()
    print_section("FESTPLATTEN INFORMATION", storage_info)
    
    # Requirements Empfehlung
    recommendations = get_requirements_recommendation(jetson_model, jetpack_version)
    print_section("🎯 EMPFOHLENE REQUIREMENTS", recommendations)
    
    # Zusammenfassung
    print(f"\n{'='*60}")
    print("📝 ZUSAMMENFASSUNG FÜR REQUIREMENTS.TXT")
    print('='*60)
    print(f"Dein System: {jetson_model}")
    print(f"JetPack: {jetpack_version}")
    
    if cuda_versions:
        print(f"CUDA: {list(cuda_versions.values())[0] if cuda_versions else 'Nicht gefunden'}")
    
    print("\n💡 NÄCHSTE SCHRITTE:")
    print("1. Verwende diese Informationen für deine requirements.txt")
    print("2. Installiere CUDA-kompatible Versionen von PyTorch")
    print("3. Stelle sicher, dass TensorRT installiert ist")
    print("4. Verwende OpenCV mit CUDA-Support für beste Performance")
    
    print(f"\n🔗 NÜTZLICHE LINKS:")
    print("- NVIDIA JetPack: https://developer.nvidia.com/jetpack")
    print("- PyTorch für Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson")
    print("- TensorRT: https://developer.nvidia.com/tensorrt")

if __name__ == "__main__":
    main()