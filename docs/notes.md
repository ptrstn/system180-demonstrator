uvicorn main:app --host 0.0.0.0 --port 8000

https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide/

Follow this guide:

https://docs.luxonis.com/hardware/platform/deploy/to-jetson/

Luxonis Blob Converter:
https://blobconverter.luxonis.com/

scp *.pt *.onnx *.blob *.engine sys180@192.168.178.50:models/


model = YOLO("demonstrator/NubsUpDown.pt")
model.model.yaml
{'nc': 2, 'scales': {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512], 'x': [1.0, 1.5, 512]}, 'backbone': [[-1, 1, 'Conv', [64, 3, 2]], [-1, 1, 'Conv', [128, 3, 2]], [-1, 2, 'C3k2', [256, False, 0.25]], [-1, 1, 'Conv', [256, 3, 2]], [-1, 2, 'C3k2', [512, False, 0.25]], [-1, 1, 'Conv', [512, 3, 2]], [-1, 2, 'C3k2', [512, True]], [-1, 1, 'Conv', [1024, 3, 2]], [-1, 2, 'C3k2', [1024, True]], [-1, 1, 'SPPF', [1024, 5]], [-1, 2, 'C2PSA', [1024]]], 'head': [[-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 6], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]], [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 4], 1, 'Concat', [1]], [-1, 2, 'C3k2', [256, False]], [-1, 1, 'Conv', [256, 3, 2]], [[-1, 13], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]], [-1, 1, 'Conv', [512, 3, 2]], [[-1, 10], 1, 'Concat', [1]], [-1, 2, 'C3k2', [1024, True]], [[16, 19, 22], 1, 'Segment', ['nc', 32, 256]]], 'scale': 's', 'yaml_file': 'yolo11s-seg.yaml', 'ch': 3}

model2 = YOLO("demonstrator/system180custommodel_v1.pt")
model2.model.yaml
{'nc': 26, 'scales': {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512], 'x': [1.0, 1.5, 512]}, 'backbone': [[-1, 1, 'Conv', [64, 3, 2]], [-1, 1, 'Conv', [128, 3, 2]], [-1, 2, 'C3k2', [256, False, 0.25]], [-1, 1, 'Conv', [256, 3, 2]], [-1, 2, 'C3k2', [512, False, 0.25]], [-1, 1, 'Conv', [512, 3, 2]], [-1, 2, 'C3k2', [512, True]], [-1, 1, 'Conv', [1024, 3, 2]], [-1, 2, 'C3k2', [1024, True]], [-1, 1, 'SPPF', [1024, 5]], [-1, 2, 'C2PSA', [1024]]], 'head': [[-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 6], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]], [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 4], 1, 'Concat', [1]], [-1, 2, 'C3k2', [256, False]], [-1, 1, 'Conv', [256, 3, 2]], [[-1, 13], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]], [-1, 1, 'Conv', [512, 3, 2]], [[-1, 10], 1, 'Concat', [1]], [-1, 2, 'C3k2', [1024, True]], [[16, 19, 22], 1, 'Detect', ['nc']]], 'scale': 's', 'yaml_file': 'yolo11s.yaml', 'ch': 3}

model3 = YOLO("demonstrator/best_synthetic_v2.pt")
model3.model.yaml
{'nc': 29, 'scales': {'n': [0.5, 0.25, 1024], 's': [0.5, 0.5, 1024], 'm': [0.5, 1.0, 512], 'l': [1.0, 1.0, 512], 'x': [1.0, 1.5, 512]}, 'backbone': [[-1, 1, 'Conv', [64, 3, 2]], [-1, 1, 'Conv', [128, 3, 2]], [-1, 2, 'C3k2', [256, False, 0.25]], [-1, 1, 'Conv', [256, 3, 2]], [-1, 2, 'C3k2', [512, False, 0.25]], [-1, 1, 'Conv', [512, 3, 2]], [-1, 2, 'C3k2', [512, True]], [-1, 1, 'Conv', [1024, 3, 2]], [-1, 2, 'C3k2', [1024, True]], [-1, 1, 'SPPF', [1024, 5]], [-1, 2, 'C2PSA', [1024]]], 'head': [[-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 6], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]], [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 4], 1, 'Concat', [1]], [-1, 2, 'C3k2', [256, False]], [-1, 1, 'Conv', [256, 3, 2]], [[-1, 13], 1, 'Concat', [1]], [-1, 2, 'C3k2', [512, False]], [-1, 1, 'Conv', [512, 3, 2]], [[-1, 10], 1, 'Concat', [1]], [-1, 2, 'C3k2', [1024, True]], [[16, 19, 22], 1, 'Detect', ['nc']]], 'scale': 'l', 'yaml_file': 'yolo11l.yaml', 'ch': 3}


Got it. Here's a clean and readable **Markdown table** comparing the important YAML parameters of the three YOLO models:

---

### 📊 YOLO Model Configuration Overview

| Model Name                   | `nc` (Classes) | `scale` | `yaml_file`        | `ch` | `head` Type | Notable Last Layer         |
|------------------------------|----------------|---------|--------------------|------|-------------|----------------------------|
| `NubsUpDown.pt`              | 2              | `s`     | `yolo11s-seg.yaml` | 3    | `Segment`   | `Segment(['nc', 32, 256])` |
| `system180custommodel_v1.pt` | 26             | `s`     | `yolo11s.yaml`     | 3    | `Detect`    | `Detect(['nc'])`           |
| `best_synthetic_v2.pt`       | 29             | `l`     | `yolo11l.yaml`     | 3    | `Detect`    | `Detect(['nc'])`           |

---

### 🔧 Scale Parameters

| Scale | `n`               | `s`              | `m`             | `l`             | `x`             |
|-------|-------------------|------------------|-----------------|-----------------|-----------------|
| All   | \[0.5, 0.25,1024] | \[0.5, 0.5,1024] | \[0.5, 1.0,512] | \[1.0, 1.0,512] | \[1.0, 1.5,512] |

*All three models use the same scale configuration.*



Install packages:

https://docs.ultralytics.com/de/guides/nvidia-jetson/#install-ultralytics-package

cat /proc/device-tree/model
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super

sudo apt-cache show nvidia-jetpack
[sudo] password for sys180: 
Package: nvidia-jetpack
Source: nvidia-jetpack (6.2)
Version: 6.2+b77
Architecture: arm64
Maintainer: NVIDIA Corporation
Installed-Size: 194
Depends: nvidia-jetpack-runtime (= 6.2+b77), nvidia-jetpack-dev (= 6.2+b77)
Homepage: http://developer.nvidia.com/jetson
Priority: standard
Section: metapackages
Filename: pool/main/n/nvidia-jetpack/nvidia-jetpack_6.2+b77_arm64.deb
Size: 29298
SHA256: 70553d4b5a802057f9436677ef8ce255db386fd3b5d24ff2c0a8ec0e485c59cd
SHA1: 9deab64d12eef0e788471e05856c84bf2a0cf6e6
MD5sum: 4db65dc36434fe1f84176843384aee23
Description: NVIDIA Jetpack Meta Package
Description-md5: ad1462289bdbc54909ae109d1d32c0a8

Package: nvidia-jetpack
Source: nvidia-jetpack (6.1)
Version: 6.1+b123
Architecture: arm64
Maintainer: NVIDIA Corporation
Installed-Size: 194
Depends: nvidia-jetpack-runtime (= 6.1+b123), nvidia-jetpack-dev (= 6.1+b123)
Homepage: http://developer.nvidia.com/jetson
Priority: standard
Section: metapackages
Filename: pool/main/n/nvidia-jetpack/nvidia-jetpack_6.1+b123_arm64.deb
Size: 29312
SHA256: b6475a6108aeabc5b16af7c102162b7c46c36361239fef6293535d05ee2c2929
SHA1: f0984a6272c8f3a70ae14cb2ca6716b8c1a09543
MD5sum: a167745e1d88a8d7597454c8003fa9a4
Description: NVIDIA Jetpack Meta Package
Description-md5: ad1462289bdbc54909ae109d1d32c0a8


Detect:
ultralytics export model=system180custommodel_v1.pt format=engine device=0 imgsz=320 half=True
-> custom_320_FP16_detect.engine
# ultralytics export model=system180custommodel_v1.pt format=engine device=0 imgsz=640 half=True
ultralytics export model=best_synthetic_v2.pt format=engine device=0 imgsz=320 half=True

Segment:
ultralytics export model=NubsUpDown.pt format=engine device=0 imgsz=320 half=True


ultralytics export model=NubsUpDown.pt format=engine device=0 imgsz=320 half=True
Ultralytics 8.3.148 🚀 Python-3.10.12 torch-2.5.0a0+872d972e41.nv24.08 CUDA:0 (Orin, 7620MiB)
YOLO11s-seg summary (fused): 113 layers, 10,067,590 parameters, 0 gradients, 35.3 GFLOPs

PyTorch: starting from 'NubsUpDown.pt' with input shape (1, 3, 320, 320) BCHW and output shape(s) ((1, 38, 2100), (1, 32, 80, 80)) (19.6 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.56...
ONNX: export success ✅ 3.7s, saved as 'NubsUpDown.onnx' (38.5 MB)

TensorRT: starting export with TensorRT 10.3.0...
[06/04/2025-10:01:08] [TRT] [I] [MemUsageChange] Init CUDA: CPU +2, GPU +0, now: CPU 828, GPU 3172 (MiB)
[06/04/2025-10:01:10] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +927, GPU +968, now: CPU 1798, GPU 4141 (MiB)
[06/04/2025-10:01:10] [TRT] [I] ----------------------------------------------------------------
[06/04/2025-10:01:10] [TRT] [I] Input filename:   NubsUpDown.onnx
[06/04/2025-10:01:10] [TRT] [I] ONNX IR version:  0.0.9
[06/04/2025-10:01:10] [TRT] [I] Opset version:    19
[06/04/2025-10:01:10] [TRT] [I] Producer name:    pytorch
[06/04/2025-10:01:10] [TRT] [I] Producer version: 2.5.0
[06/04/2025-10:01:10] [TRT] [I] Domain:           
[06/04/2025-10:01:10] [TRT] [I] Model version:    0
[06/04/2025-10:01:10] [TRT] [I] Doc string:       
[06/04/2025-10:01:10] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 320, 320) DataType.FLOAT
TensorRT: output "output0" with shape(1, 38, 2100) DataType.FLOAT
TensorRT: output "output1" with shape(1, 32, 80, 80) DataType.FLOAT
TensorRT: building FP16 engine as NubsUpDown.engine
[06/04/2025-10:01:10] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[06/04/2025-10:08:57] [TRT] [I] Detected 1 inputs and 5 output network tensors.
[06/04/2025-10:09:02] [TRT] [I] Total Host Persistent Memory: 568416
[06/04/2025-10:09:02] [TRT] [I] Total Device Persistent Memory: 30720
[06/04/2025-10:09:02] [TRT] [I] Total Scratch Memory: 211456
[06/04/2025-10:09:02] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 194 steps to complete.
[06/04/2025-10:09:02] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 31.7864ms to assign 12 blocks to 194 nodes requiring 5436416 bytes.
[06/04/2025-10:09:02] [TRT] [I] Total Activation Memory: 5434880
[06/04/2025-10:09:03] [TRT] [I] Total Weights Memory: 20195976
[06/04/2025-10:09:03] [TRT] [I] Engine generation completed in 472.537 seconds.
[06/04/2025-10:09:03] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 48 MiB
[06/04/2025-10:09:03] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 2881 MiB
TensorRT: export success ✅ 479.3s, saved as 'NubsUpDown.engine' (22.7 MB)

Export complete (481.1s)
Results saved to /home/sys180/models
Predict:         yolo predict task=segment model=NubsUpDown.engine imgsz=320 half 
Validate:        yolo val task=segment model=NubsUpDown.engine imgsz=320 data=/content/datasets/NubsDetection-2/data.yaml half 
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export

