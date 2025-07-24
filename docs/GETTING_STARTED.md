# 🎯 Getting Started with System180 Demonstrator

**For Non-Programmers** - A simple step-by-step guide

---

## 📋 What You Need

Before starting, make sure you have:
- A computer with **Python 3** installed
- Your **3 model files** (custom.pt, NubsUpDown.pt, synthetic.pt)
- Your **cameras** connected to the computer
- An **internet connection** (for downloading packages)

---

## 🚀 Quick Start (3 Easy Steps)

### Step 1: Setup Everything
```bash
bash setup.sh
```
**What this does:** Installs all the software packages needed to run the system.

### Step 2: Convert Your Models
```bash
bash convert_models_to_engine.sh
```
**What this does:** Converts your AI models to a faster format for your computer.
**⏱️ Time:** This takes about 5-10 minutes per model.

### Step 3: Start the Application
```bash
bash run.sh
```
**What this does:** Starts the web application that shows the camera feeds.
**🌐 Access:** Open your web browser and go to http://localhost:8000

---

## 📁 File Organization

Put your files in the right places:

```
your-project-folder/
├── models/                  ← PUT YOUR .pt MODEL FILES HERE
│   ├── custom.pt           ← Your main detection model
│   ├── NubsUpDown.pt       ← Your segmentation model
│   └── synthetic.pt        ← Your synthetic data model
├── setup.sh                ← Script 1: Install everything
├── convert_models_to_engine.sh ← Script 2: Convert models
└── run.sh                  ← Script 3: Start the app
```

---

## 🎥 Camera Setup

The system expects **3 cameras**:
- **Left Camera:** OAK-1 Max (for nub detection)
- **Right Camera:** OAK-1 Max (for nub detection)  
- **Center Camera:** OBSBOT Meet 2 or similar USB camera (for defect detection)

Make sure all cameras are **connected** before running the application.

---

## ❓ Troubleshooting

### Problem: "Python3 not found"
**Solution:** Install Python 3 from https://python.org

### Problem: "Models not found"
**Solution:** Make sure your .pt files are in the `models/` folder with the exact names:
- `custom.pt`
- `NubsUpDown.pt`
- `synthetic.pt`

### Problem: "Port 8000 already in use"
**Solution:** The application will automatically try other ports (8001, 8002, etc.)

### Problem: Camera not working
**Solution:** 
1. Check camera connections
2. Try unplugging and plugging back in
3. Restart the application with `bash run.sh`

### Problem: Models conversion failed
**Solution:**
1. Make sure you have a NVIDIA GPU
2. Check that CUDA is installed
3. Verify your .pt model files are not corrupted

---

## 🛑 How to Stop the Application

Press **Ctrl+C** in the terminal window where the application is running.

---

## 📞 Getting Help

If you encounter problems:
1. Check the **terminal output** for error messages
2. Make sure you followed each step in order
3. Verify all files are in the correct locations
4. Ask your technical contact for help with the error message

---

## 🤖 Jetson System Check

**Neu!** Bevor du anfängst, prüfe dein Jetson System:

```bash
# Schnelle System-Info (Bash)
bash jetson_info.sh

# Detaillierte Analyse (Python)
python3 jetson_info.py
```

**Was das macht:** Zeigt dir JetPack Version, CUDA, TensorRT etc. an, damit du die richtigen Requirements installierst.

---

## 📹 Kamera Setup

**Problem:** Du weißt nicht welche OAK-Kamera links/rechts ist?

```bash
# Kameras identifizieren (mit Live-Vorschau!)
python3 find_cameras.py

# Kameras tauschen falls nötig
python3 swap_cameras.py
```

**Mehr Details:** Siehe `KAMERA_SETUP.md`

---

## 🔧 Advanced Options

### Running on a Different Port
If you need to use a specific port:
```bash
python3 main.py --port 8080
```

### Production Mode (Faster)
For better performance:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

**✅ That's it! You should now have the System180 Demonstrator running successfully.**