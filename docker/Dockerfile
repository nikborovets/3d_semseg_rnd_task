FROM pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel
WORKDIR /workspace

ARG CUDA_ARCH="7.5 8.0 8.6"
ARG MAX_JOBS=14

ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    MAX_JOBS=${MAX_JOBS}

# для сборки MinkowskiEngine
RUN apt-get update && apt-get install -y git ninja-build cmake build-essential libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install open3d

COPY third_party/MinkowskiEngine /workspace/MinkowskiEngine
COPY third_party/KPConv-PyTorch /workspace/KPConv-PyTorch

RUN cd /workspace/MinkowskiEngine && python setup.py install --force_cuda --blas=openblas
RUN cd /workspace/KPConv-PyTorch/cpp_wrappers && bash compile_wrappers.sh
RUN rm -rf /workspace/MinkowskiEngine && rm -rf /workspace/KPConv-PyTorch
