# https://docs.ultralytics.com/de/guides/nvidia-jetson/#start-with-native-installation
sudo apt update
sudo apt install python3-pip -y
pip install -U pip
pip install ultralytics
pip uninstall torch torchvision
# https://docs.ultralytics.com/de/guides/nvidia-jetson/#install-pytorch-and-torchvision
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
# https://docs.ultralytics.com/de/guides/nvidia-jetson/#install-onnxruntime-gpu
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
# depthai, fastapi, ...
pip install -r requirements.txt
