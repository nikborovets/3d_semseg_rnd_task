FROM pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel
WORKDIR /workspace

# # Автоматическое определение архитектуры GPU с проверкой CUDA
# RUN CUDA_ARCH=$(python3 -c "import torch;print('.'.join(map(str,torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else 'no-cuda')") && \
#     if [ "$CUDA_ARCH" = "no-cuda" ]; then \
#         echo "Ошибка: CUDA недоступна! MinkowskiEngine требует CUDA для компиляции."; \
#         exit 1; \
#     fi && \
#     echo "Обнаружена архитектура CUDA: $CUDA_ARCH"

# ENV TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" \
# ENV TORCH_CUDA_ARCH_LIST="8.0" \
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    MAX_JOBS=8

# для сборки MinkowskiEngine
RUN apt-get update && apt-get install -y git ninja-build cmake build-essential libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install open3d

COPY third_party/MinkowskiEngine /workspace/MinkowskiEngine
COPY third_party/KPConv-PyTorch /workspace/KPConv-PyTorch

RUN cd /workspace/MinkowskiEngine && python setup.py install --force_cuda --blas=openblas
RUN cd /workspace/KPConv-PyTorch/cpp_wrappers && bash compile_wrappers.sh
RUN rm -rf /workspace/MinkowskiEngine && rm -rf /workspace/KPConv-PyTorch