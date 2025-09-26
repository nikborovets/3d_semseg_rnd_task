# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import sys
import os
import argparse
import numpy as np
from urllib.request import urlretrieve
import time
import open3d as o3d
import torch

from utils.utils import set_random_seed

sys.path.append("./third_party/MinkowskiEngine")
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C


CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

VALID_CLASS_IDS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
]

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


def parse_args():
    # --- Default Inference Parameters ---
    pcd_path = "/workspace/pcd_files/down0.01.pcd"
    weights_path = "weights.pth"
    output_dir = "result_plys/minkowski_plys"
    device = "cuda"
    voxel_size = 0.03
    seed = 42
    # ------------------------------------

    parser = argparse.ArgumentParser(
        description="MinkowskiEngine Semantic Segmentation Inference"
    )
    parser.add_argument(
        "--pcd_path", type=str, default=pcd_path, help="Path to the input PCD file."
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=weights_path,
        help="Path to the model weights file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Directory to save the output PLY file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=device,
        choices=["cuda", "cpu"],
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=voxel_size,
        help="Voxel size for quantization.",
    )
    parser.add_argument(
        "--seed", type=int, default=seed, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def download_assets():
    # Check if the weights and file exist and download
    if not os.path.isfile("weights.pth"):
        print("Downloading weights...")
        urlretrieve("https://bit.ly/2O4dZrz", "weights.pth")
    # if not os.path.isfile("1.ply"):
    #     print('Downloading an example pointcloud...')
    #     urlretrieve("https://bit.ly/3c2iLhg", "1.ply")


def load_pcd(file_path):
    print(f"Loading point cloud from {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors) if pcd.has_colors() else np.ones_like(coords) * 0.5
    return coords, colors, pcd


def normalize_color(colors):
    """Normalizes colors to the range [-0.5, 0.5]"""
    return torch.from_numpy(colors).float() - 0.5


class MinkowskiInferencer:
    def __init__(self, weights_path, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = MinkUNet34C(3, 20).to(self.device)
        print("Loading model weights...")
        model_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        print("Model loaded successfully!")

    def predict(self, coords, colors, voxel_size):
        print("Performing inference...")
        with torch.no_grad():
            # Prepare input data
            in_field = ME.TensorField(
                features=normalize_color(colors),
                coordinates=ME.utils.batched_coordinates(
                    [coords / voxel_size], dtype=torch.float32
                ),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
            sinput = in_field.sparse()

            # Forward pass
            start_time = time.time()
            soutput = self.model(sinput)
            inference_time = time.time() - start_time
            print(f"  Inference time: {inference_time:.2f} seconds")

            logits = soutput.F
            _, pred = logits.max(1)

            voxel_coords = soutput.C[:, 1:].cpu().numpy() * voxel_size
            predictions = pred.cpu().numpy()

        print(f"Processed {len(voxel_coords)} voxels (from {len(coords)} points)")
        return voxel_coords, predictions


def main():
    args = parse_args()

    set_random_seed(args.seed)

    print("=" * 80)
    print("MINKOWSKI ENGINE INFERENCE - SEMANTIC SEGMENTATION")
    print("=" * 80)
    print(f"Input file: {args.pcd_path}")
    print(f"Model: MinkUNet34C (weights: {args.weights_path})")
    print(f"Parameters: voxel_size={args.voxel_size}m, seed={args.seed}")

    download_assets()

    try:
        # 1. Load data
        coords, colors, _ = load_pcd(args.pcd_path)

        # 2. Initialize model
        inferencer = MinkowskiInferencer(args.weights_path, args.device)

        # 3. Perform inference
        voxel_coords, predictions = inferencer.predict(coords, colors, args.voxel_size)

        # 4. Visualize and save
        print("\n4. Creating colored point cloud...")
        pred_pcd = o3d.geometry.PointCloud()
        pred_colors = np.array(
            [SCANNET_COLOR_MAP[VALID_CLASS_IDS[label]] for label in predictions]
        )

        pred_pcd.points = o3d.utility.Vector3dVector(voxel_coords)
        pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors / 255)

        # Create output filename
        input_filename = os.path.splitext(os.path.basename(args.pcd_path))[0]
        output_filename = (
            f"{input_filename}_Minkowski_MinkUNet34C_"
            f"voxel{args.voxel_size}m_"
            f"segmented_seed_{args.seed}.ply"
        )
        output_path = os.path.join(args.output_dir, output_filename)

        print(f"\n5. Saving result to {output_path}...")
        os.makedirs(args.output_dir, exist_ok=True)
        o3d.io.write_point_cloud(output_path, pred_pcd)

        print("\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"   Processed points: {len(coords)}")
        print(f"   Result saved to: {output_path}")
        print(f"   Classes found: {len(np.unique(predictions))}")

        # Class statistics
        unique_classes, counts = np.unique(predictions, return_counts=True)
        print("\nClass statistics:")
        for class_id, count in zip(unique_classes, counts):
            class_name = CLASS_LABELS[class_id]
            percentage = (count / len(predictions)) * 100
            print(f"  {class_name}: {count} voxels ({percentage:.1f}%)")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
