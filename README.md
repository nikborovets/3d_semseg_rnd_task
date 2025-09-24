# 3d_semseg_rnd_task

### Установка Docker и NVIDIA Container Toolkit

# 1. Установка Docker
sudo apt update
apt install git unzip htop -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh

# 2. Установка NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Настройка Docker для использования NVIDIA runtime и перезапуск
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


docker pull pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel
git clone https://github.com/nikborovets/3d_semseg_rnd_task.git

# export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
make -f all

cd Pointcept
docker compose up -d

docker compose exec pointcept_me /bin/bash
cd third_party/KPConv-PyTorch/cpp_wrappers && bash compile_wrappers.sh && cd ../../../

python mink_inference.py
python sonata_inference.py
python kpconv_inference.py
```




```bash
# set -e

git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept

mkdir -p third_party && cd third_party

git clone --recursive https://github.com/NVIDIA/MinkowskiEngine.git
git clone https://github.com/facebookresearch/sonata.git
git clone https://github.com/HuguesTHOMAS/KPConv-PyTorch.git

cd ..
cp ../docker/Dockerfile .
cp ../docker/docker-compose.yml .

docker compose build pointcept
docker compose run --rm pointcept python -c "import torch, MinkowskiEngine as ME; print(torch.__version__, ME.__version__)"
# зайти в контейнер и запустить демонстрацию
# docker exec -it pointcept_me /bin/bash
# export PYTHONPATH=./third_party/MinkowskiEngine && python -m examples.indoor
# export PYTHONPATH=./third_party/sonata && python -m demo.2_sem_seg
python indoor.py
python 2_sem_seg.py
python kpconv_inference.py



wget https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip
unzip Stanford3dDataset_v1.2_Aligned_Version.zip -d S3DIS
# RUN sed -i 's|#include <thrust/count.h>|#include <thrust/count.h>\n#include <thrust/execution_policy.h>|' /workspace/MinkowskiEngine/src/3rdparty/concurrent_unordered_map.cuh && \
#     sed -i '1i #include <thrust/execution_policy.h>' /workspace/MinkowskiEngine/src/spmm.cu && \
#     sed -i '1i #include <thrust/remove.h>\n#include <thrust/unique.h>\n#include <thrust/for_each.h>\n#include <thrust/transform.h>\n#include <thrust/sequence.h>\n#include <thrust/scan.h>' /workspace/MinkowskiEngine/src/coordinate_map_gpu.cu
```