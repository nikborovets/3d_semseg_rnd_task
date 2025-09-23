FROM pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel
WORKDIR /workspace

ENV TORCH_CUDA_ARCH_LIST="7.5" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    MAX_JOBS=8

# для сборки MinkowskiEngine
RUN apt-get update && apt-get install -y git ninja-build cmake build-essential libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install open3d

COPY third_party/MinkowskiEngine /workspace/MinkowskiEngine
RUN cd /workspace/MinkowskiEngine && python setup.py install --force_cuda --blas=openblas
RUN rm -rf /workspace/MinkowskiEngine