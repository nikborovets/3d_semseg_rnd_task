# 3d_semseg_rnd_task

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
```