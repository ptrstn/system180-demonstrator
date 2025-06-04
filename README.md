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

