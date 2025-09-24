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
sys.path.append('./third_party/sonata')
import sonata
import torch
import torch.nn as nn

from point_analyzer import analyze_point_dict, get_point_summary
from pcd_preprocessor import load_and_preprocess_pcd


def set_random_seed(seed=42):
    """
    Устанавливает random seed для воспроизводимости результатов
    """
    print(f"🎲 Установка random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Для multi-GPU
    
    # Для полной детерминированности (может замедлить работу)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # SONATA также использует random seed
    try:
        sonata.utils.set_seed(seed)
        print("✅ SONATA seed установлен")
    except:
        print("⚠️  SONATA seed не установлен (модуль не найден)")
    
    print("✅ Random seed установлен для всех библиотек")

random_seed = 42
set_random_seed(random_seed)
# try:
#     import flash_attn
# except ImportError:
#     flash_attn = None

flash_attn = None

import gc

# Очистка памяти перед началом
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


if __name__ == "__main__":
    # set random seed
    # sonata.utils.set_seed(24525867)
    # Load model
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        custom_config = dict(
            enc_patch_size=[512 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()
    # Load linear probing seg head
    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    # Load default data transform pipeline
    transform = sonata.transform.default()
    # Load data - выбор между sample1 и собственным PCD файлом
    use_sample_data = False  # Установите True для использования sample1, False для собственного PCD
    
    if use_sample_data:
        point = sonata.data.load("sample1")
        print("keys:", point.keys())
        
        analyze_point_dict(point)
        
        if "color" not in point:
            point["color"] = np.ones((len(point["coord"]), 3)) * 0.5
        
        if "normal" not in point:
            print("Computing normals for sample1_dino...")
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(point["coord"])
            pcd_temp.estimate_normals()
            point["normal"] = np.array(pcd_temp.normals)
        
        if "segment20" not in point and "segment200" not in point:
            print("Adding dummy segmentation for sample1_dino")
            point["segment"] = np.zeros(len(point["coord"]), dtype=np.int32)
        else:
            # Обрабатываем существующую сегментацию
            if "segment200" in point:
                point.pop("segment200")
            if "segment20" in point:
                segment = point.pop("segment20")
                point["segment"] = segment
        # # Load data
        # point.pop("segment200")
        # segment = point.pop("segment20")
        # point["segment"] = segment  # two kinds of segment exist in ScanNet, only use one

        # # Детальный анализ всех параметров словаря point
        # # analyze_point_dict(point)
        # # print(get_point_summary(point))
    else:
        # Используем собственный PCD файл с предобработкой
        pcd_file_path = "/workspace/pcd_files/down0.01.pcd"
        # pcd_file_path = "/workspace/pcd_files/music_room.pcd"
        
        # Загружаем и предобрабатываем PCD файл
        downsampling_method="grid"
        voxel_size=0.03  # размер вокселя в метрах (2.5 см)

        point = load_and_preprocess_pcd(
            file_path=pcd_file_path,
            # downsampling_method="voxel",  # "voxel" или "random" или "grid"
            downsampling_method=downsampling_method,  # "voxel" или "random" или "grid"
            voxel_size=voxel_size,  # размер вокселя в метрах (2.5 см)
            target_points=193982,  # для random downsampling
            add_segmentation=True
        )
        
        # Анализируем предобработанные данные
        # analyze_point_dict(point)
        # for key, value in get_point_summary(point).items():
        #     print("{0}: {1}".format(key,value))
    
    gc.collect()
    # original_coord = point["coord"].copy()
    point = transform(point)
    gc.collect()
    # Inference
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]

    # Статистика по классам
    print(f"\n--- Статистика классов ---")
    unique_classes, counts = np.unique(pred, return_counts=True)
    for class_id, count in zip(unique_classes, counts):
        if class_id < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[class_id]
        else:
            class_name = "unknown"
        percentage = (count / len(pred)) * 100
        print(f"   {class_name}: {count} точек ({percentage:.1f}%)")

    del seg_logits, pred
    gc.collect()
    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color / 255)

    # Создаем динамическое имя выходного файла
    input_filename = os.path.splitext(os.path.basename(pcd_file_path))[0]
    model_name = "sonata"
    output_filename = (f"{input_filename}_Sonata_{model_name}_"
                      f"downsample_{downsampling_method}_voxel{voxel_size}m_"
                      f"segmented_seed_{random_seed}.ply")
    output_path = f"result_plys/sonata_plys/{output_filename}"

    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("sem_seg_new_256_0025.ply", pcd)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
