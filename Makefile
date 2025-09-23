.PHONY: download_pcd download_weights build_docker all

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
	@echo "Сборка Docker образа..."
	cd Pointcept && docker compose build

all: download_pcd download_weights build_docker
	@echo "Все задачи выполнены. Файлы скачаны, Docker образ собран."
