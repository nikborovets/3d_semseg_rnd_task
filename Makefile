.PHONY: all build_docker download_data download_pcd download_weights

CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)

# make -j all

download_pcd:
	@echo "Скачивание PCD файлов..."
	@export ARCHIVE_FILE=yandex_disk_archive.zip && \
	 export PUBLIC_KEY_URL=https://disk.360.yandex.ru/d/-oVuujK-B4LryQ && \
	 export META_DIR=pcd_files && \
	 ./download_and_extract_pcd.sh

download_weights:
	@echo "Скачивание KPConv weights..."
	@export ARCHIVE_FILE=kpconv_weights_yandex_disk.zip && \
	 export PUBLIC_KEY_URL=https://disk.yandex.ru/d/eX34iBhLON7wXg && \
	 export META_DIR=kpconv_weights/Light_KPFCNN && \
	 ./download_and_extract_pcd.sh

build_docker:
ifeq ($(CUDA_ARCH),)
	$(error "Не удалось определить архитектуру CUDA. Убедитесь, что nvidia-smi работает и драйверы NVIDIA в порядке.")
endif
	@echo "Сборка Docker образа для архитектуры CUDA: ${CUDA_ARCH}"
	export CUDA_ARCH=${CUDA_ARCH} && \
	 docker compose build

download_data: download_pcd download_weights

all: download_data build_docker
	@echo "Все задачи выполнены. Файлы скачаны, Docker образ собран."
