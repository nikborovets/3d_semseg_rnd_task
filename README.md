# 3d_semseg_rnd_task

### 1. Установка Docker
```bash
sudo apt update
apt install git unzip htop -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh
```
### 2. Установка NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
### 3. Настройка Docker для использования NVIDIA runtime и перезапуск
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Скачивание Pointcept и данных
```bash
docker pull pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel
git clone https://github.com/nikborovets/3d_semseg_rnd_task.git
cd 3d_semseg_rnd_task
```

### Сборка контейнера
```bash
make -f all
# or if you want to download data and build docker separately
# make download_data
# make build_docker

docker compose up -d

# docker compose exec pointcept_me /bin/bash
docker exec -it 3d_semseg_rnd_task-pointcept_me-1 /bin/bash
cd Pointcept/third_party/KPConv-PyTorch/cpp_wrappers && bash compile_wrappers.sh && cd ../../../../

python mink_inference.py
python sonata_inference.py
python kpconv_inference.py

python mink_inference.py \
    --voxel_size 0.03

python sonata_inference.py \
    --downsampling_method "grid" \
    --voxel_size 0.03 \
    --enc_patch_size 512

python kpconv_inference.py \
    --downsampling_method "grid" \
    --voxel_size 0.03 \
    --chunk_size 300000

```
