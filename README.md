# 3D Semantic Segmentation for Dense Point Clouds

This repository contains a reproducible pipeline for semantic segmentation of dense, colored indoor point clouds using three state-of-the-art models (**KPConv**, **MinkowskiNet**, and **Sonata**). It is submitted as a solution for the R&D Assignment Problem from the Artificial Intelligence for Autonomous Systems Laboratory.

### Example Visualization

Below is an example of segmentation results from the Sonata model:

<table>
  <tr>
    <td style="width: 50%; vertical-align: top; padding-right: 20px;">
      <p><strong>Sonata Model Results:</strong></p>
      <p>This visualization demonstrates the semantic segmentation capabilities of the Sonata model on dense indoor point clouds. The model successfully identifies various semantic classes such as walls, floors, furniture, and other objects with high accuracy.</p>
      <p><em>Note:</em> The segmentation preserves the original point cloud structure while assigning semantic labels to each point.</p>
    </td>
    <td style="width: 50%; text-align: center;">
      <img src="./assets/small_sonata_512_003.gif" alt="Sonata Segmentation Result" width="250"/>
    </td>
  </tr>
</table>

## Prerequisites

Before you begin, ensure you have the following installed on your host machine:
- Docker
- NVIDIA Container Toolkit for GPU support

### 1. Docker Installation
```bash
sudo apt update
sudo apt install git unzip htop -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh
```

### 2. NVIDIA Container Toolkit Installation
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit
```

### 3. Docker Configuration
Configure Docker to use the NVIDIA runtime and restart the service.
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Getting Started

Follow these steps to set up the project environment and run the inference pipeline.

### Step 1: Clone the Repository and Download Dependencies
First, pull the required Docker image and clone this repository.
```bash
docker pull pointcept/pointcept:v1.5.0-pytorch1.11.0-cuda11.3-cudnn8-devel
git clone https://github.com/nikborovets/3d_semseg_rnd_task.git
cd 3d_semseg_rnd_task
```

### Step 2: Build the Environment and Download Data
Use the provided `Makefile` to build the Docker container and download the necessary data.
```bash
# This command will download data and build the docker image
make -j all

# Alternatively, you can run these steps separately
# make download_data
# make build_docker
```

### Step 3: Start the Container
Run the Docker container in detached mode.
```bash
docker compose up -d
```

### Step 4: Post-build Setup
Enter the running container to compile C++/CUDA wrappers required by KPConv.
```bash
docker exec -it 3d_semseg_rnd_task-pointcept_me-1 /bin/bash

# Inside the container, run the following commands:
cd Pointcept/third_party/KPConv-PyTorch/cpp_wrappers
bash compile_wrappers.sh
cd ../../../../
```

## Usage

Once the setup is complete, you can run inference using the provided scripts.

### Running Inference with Default Parameters
Execute the following scripts to run segmentation with the default model configurations.
```bash
python mink_inference.py
python sonata_inference.py
python kpconv_inference.py
```

### Running Inference with Custom Parameters
You can also run the inference scripts with custom arguments to adjust the models' behavior.
```bash
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

## Outputs

The inference scripts save the segmented point clouds as `.ply` files in the `result_plys/` directory. The results for each model are stored in separate subdirectories:

-   **KPConv:** `result_plys/kpconv_plys/`
-   **MinkowskiNet:** `result_plys/minkowski_plys/`
-   **Sonata:** `result_plys/sonata_plys/`

The output filenames are structured to be descriptive and include the model name, input file, and key inference parameters.

**Example Filename Format:**
```
<input_filename>_<model_name>_<parameters>_segmented_seed_<seed_number>.ply
```

**Concrete Example (from KPConv):**
```
down0.01_KPConv_Light_KPFCNN_downsample_grid_voxel0.03m_chunk300000_segmented_seed_42.ply
```
