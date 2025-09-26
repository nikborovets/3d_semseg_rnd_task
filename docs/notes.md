
#### Заметки
```bash
# set -e

git submodule add https://github.com/NVIDIA/MinkowskiEngine.git third_party/MinkowskiEngine && cd third_party/MinkowskiEngine && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && cd ../../

git submodule add https://github.com/facebookresearch/sonata.git third_party/sonata && cd third_party/sonata && git checkout 18c09ff8d713494f78a8213792262b910977a65d && cd ../../

git submodule add https://github.com/HuguesTHOMAS/KPConv-PyTorch.git third_party/KPConv-PyTorch && cd third_party/KPConv-PyTorch && git checkout bf19211adf52098a90617ef21a677745d9775b10 && cd ../../


cd ../../

docker compose build
docker compose run --rm python -c "import torch, MinkowskiEngine as ME; print(torch.__version__, ME.__version__)"
# зайти в контейнер и запустить демонстрацию
# docker exec -it pointcept_me /bin/bash
# export PYTHONPATH=./third_party/MinkowskiEngine && python -m examples.indoor
# export PYTHONPATH=./third_party/sonata && python -m demo.2_sem_seg
python mink_inference.py
python sonata_inference.py
python kpconv_inference.py



wget https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip
unzip Stanford3dDataset_v1.2_Aligned_Version.zip -d S3DIS
# RUN sed -i 's|#include <thrust/count.h>|#include <thrust/count.h>\n#include <thrust/execution_policy.h>|' /workspace/MinkowskiEngine/src/3rdparty/concurrent_unordered_map.cuh && \
#     sed -i '1i #include <thrust/execution_policy.h>' /workspace/MinkowskiEngine/src/spmm.cu && \
#     sed -i '1i #include <thrust/remove.h>\n#include <thrust/unique.h>\n#include <thrust/for_each.h>\n#include <thrust/transform.h>\n#include <thrust/sequence.h>\n#include <thrust/scan.h>' /workspace/MinkowskiEngine/src/coordinate_map_gpu.cu
```