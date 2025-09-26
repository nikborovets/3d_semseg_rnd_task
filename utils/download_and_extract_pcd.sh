#!/bin/bash

# chmod +x download_and_extract_pcd.sh

# ARCHIVE_FILE="yandex_disk_archive.zip"
# PUBLIC_KEY_URL="https://disk.360.yandex.ru/d/-oVuujK-B4LryQ"
# META_DIR="pcd_files"


# ARCHIVE_FILE="kpconv_weights_yandex_disk.zip"
# PUBLIC_KEY_URL="https://disk.yandex.ru/d/eX34iBhLON7wXg"
# META_DIR="kpconv_weights/Light_KPFCNN"


if ! command -v jq &> /dev/null
then
    echo "jq not found. Attempting to install..."
    if command -v brew &> /dev/null; then
        brew install jq
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y jq
    else
        echo "jq is required but could not be installed automatically. Please install jq and retry."
        exit 1
    fi
fi

DOWNLOAD_LINK=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=${PUBLIC_KEY_URL}" | jq -r '.href')

if [ -z "$DOWNLOAD_LINK" ]; then
    echo "Failed to get download link. Exiting."
    exit 1
fi

if [ -f $ARCHIVE_FILE ]; then
    echo "$ARCHIVE_FILE already exists. Skipping download."
else
    echo "Downloading yandex_disk_archive.zip..."
    wget -O $ARCHIVE_FILE "$DOWNLOAD_LINK"

    if [ $? -ne 0 ]; then
        echo "Failed to download $ARCHIVE_FILE. Exiting."
        exit 1
    fi
fi

mkdir -p temp_extraction_folder

unzip "$ARCHIVE_FILE" -d temp_extraction_folder

if [ $? -ne 0 ]; then
    rm -rf temp_extraction_folder
    exit 1
fi

mkdir -p $META_DIR

EXTRACTED_FOLDER_FULL_PATH=$(find temp_extraction_folder -mindepth 1 -maxdepth 1 -type d -print -quit)
EXTRACTED_FOLDER_NAME=$(basename "$EXTRACTED_FOLDER_FULL_PATH")

if [ -z "$EXTRACTED_FOLDER_NAME" ]; then
    echo "Could not find extracted folder inside temp_extraction_folder. Exiting."
    rm -rf temp_extraction_folder
    exit 1
fi

mv "temp_extraction_folder/$EXTRACTED_FOLDER_NAME"/* $META_DIR/

if [ $? -ne 0 ]; then
    echo "Failed to move files to $META_DIR. Exiting."
    rm -rf temp_extraction_folder
    exit 1
fi

echo "Checking for nested zip files in $META_DIR..."
NESTED_ZIP_FILES=$(find $META_DIR -name "*.zip")

if [ -n "$NESTED_ZIP_FILES" ]; then
    for zip_file in $NESTED_ZIP_FILES; do
        echo "Extracting nested zip: $zip_file"
        unzip "$zip_file" -d $META_DIR/
        if [ $? -ne 0 ]; then
            echo "Failed to extract nested zip: $zip_file. Continuing with other files."
        # else
        #     rm "$zip_file"
        fi
    done
fi

rm -rf temp_extraction_folder

echo "PCD files are now available in the '$META_DIR' directory."
