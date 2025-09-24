# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import random
import numpy as np
import open3d as o3d
import argparse
import time
sys.path.append('./third_party/sonata')
import sonata
import torch
import torch.nn as nn

from point_analyzer import analyze_point_dict, get_point_summary
from pcd_preprocessor import load_and_preprocess_pcd


def parse_args():
    # --- Default Inference Parameters ---
    pcd_path = '/workspace/pcd_files/down0.01.pcd'
    output_dir = 'result_plys/sonata_plys'
    device = 'cuda'
    downsampling_method = 'grid'
    voxel_size = 0.03
    seed = 42
    enc_patch_size = 512
    # ------------------------------------

    parser = argparse.ArgumentParser(description="SONATA Semantic Segmentation Inference")
    parser.add_argument('--pcd_path', type=str, default=pcd_path,
                        help='Path to the input PCD file.')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='Directory to save the output PLY file.')
    parser.add_argument('--device', type=str, default=device, choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
    parser.add_argument('--downsampling_method', type=str, default=downsampling_method, choices=['grid', 'voxel', 'random'],
                        help='Downsampling method.')
    parser.add_argument('--voxel_size', type=float, default=voxel_size,
                        help='Voxel size for downsampling.')
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed for reproducibility.')
    parser.add_argument('--enc_patch_size', type=int, default=enc_patch_size,
                        help='Encoder patch size for SONATA model.')
    return parser.parse_args()


def set_random_seed(seed=42):
    """
    Sets the random seed for reproducibility
    """
    print(f"🎲 Setting random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # For full determinism (can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # SONATA also uses a random seed
    try:
        sonata.utils.set_seed(seed)
        print("✅ SONATA seed set")
    except:
        print("⚠️  SONATA seed not set (module not found)")
    
    print("✅ Random seed set for all libraries")


# try:
#     import flash_attn
# except ImportError:
#     flash_attn = None

flash_attn = None

import gc

# Clear memory before starting
torch.cuda.empty_cache()
gc.collect()
print("Memory cleared")


# ScanNet Meta data
VALID_CLASS_IDS_20 = (
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
)


CLASS_LABELS_20 = (
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

SCANNET_COLOR_MAP_20 = {
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

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


class SonataInferencer:
    def __init__(self, device, enc_patch_size=512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Loading SONATA model...")
        if flash_attn is not None:
            self.model = sonata.load("sonata", repo_id="facebook/sonata").to(self.device)
        else:
            custom_config = dict(
                enc_patch_size=[enc_patch_size for _ in range(5)],
                enable_flash=False,
            )
            self.model = sonata.load(
                "sonata", repo_id="facebook/sonata", custom_config=custom_config
            ).to(self.device)
        
        print("Loading segmentation head...")
        ckpt = sonata.load(
            "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
        )
        self.seg_head = SegHead(**ckpt["config"]).to(self.device)
        self.seg_head.load_state_dict(ckpt["state_dict"])
        
        self.transform = sonata.transform.default()
        
        self.model.eval()
        self.seg_head.eval()
        print("Model and segmentation head loaded successfully!")

    def predict(self, point_dict):
        print("Performing inference...")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        start_time = time.time()
        
        point = self.transform(point_dict)
        
        with torch.inference_mode():
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].to(self.device, non_blocking=True)
            
            point = self.model(point)
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            
            feat = point.feat
            seg_logits = self.seg_head(feat)
            predictions = seg_logits.argmax(dim=-1).data.cpu().numpy()
        
        inference_time = time.time() - start_time
        print(f"  Inference time: {inference_time:.2f} seconds")
        
        coords = point.coord.cpu().detach().numpy()
        
        return coords, predictions


def main():
    args = parse_args()
    
    set_random_seed(args.seed)
    
    print("=" * 80)
    print("SONATA INFERENCE - SEMANTIC SEGMENTATION")
    print("=" * 80)
    print(f"Input file: {args.pcd_path}")
    print(f"Model: SONATA")
    print(f"Parameters: downsampling={args.downsampling_method}, voxel_size={args.voxel_size}m, seed={args.seed}, enc_patch_size={args.enc_patch_size}")

    try:
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing PCD file...")
        point_dict = load_and_preprocess_pcd(
            file_path=args.pcd_path,
            downsampling_method=args.downsampling_method,
            voxel_size=args.voxel_size,
            add_segmentation=True
        )
        
        # 2. Initialize model
        print("\n2. Initializing model...")
        inferencer = SonataInferencer(args.device, args.enc_patch_size)
        
        # 3. Perform semantic segmentation
        print("\n3. Performing semantic segmentation...")
        coords, predictions = inferencer.predict(point_dict)
        
        # 4. Visualize and save
        print("\n4. Creating colored point cloud...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(np.array(CLASS_COLOR_20)[predictions] / 255)
        
        # Create output filename
        input_filename = os.path.splitext(os.path.basename(args.pcd_path))[0]
        model_name = "sonata"
        output_filename = (f"{input_filename}_Sonata_{model_name}_"
                           f"downsample_{args.downsampling_method}_voxel{args.voxel_size}m_"
                           f"segmented_seed_{args.seed}.ply")
        output_path = os.path.join(args.output_dir, output_filename)
        
        print(f"\n5. Saving result to {output_path}...")
        os.makedirs(args.output_dir, exist_ok=True)
        o3d.io.write_point_cloud(output_path, pcd)
        
        print("\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"   Processed points: {len(coords)}")
        print(f"   Result saved to: {output_path}")
        print(f"   Classes found: {len(np.unique(predictions))}")

        # Class statistics
        print(f"\n--- Class Statistics ---")
        unique_classes, counts = np.unique(predictions, return_counts=True)
        for class_id, count in zip(unique_classes, counts):
            class_name = CLASS_LABELS_20[class_id] if class_id < len(CLASS_LABELS_20) else "unknown"
            percentage = (count / len(predictions)) * 100
            print(f"   {class_name}: {count} points ({percentage:.1f}%)")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
