#!/usr/bin/env python3
"""
Script for KPConv model inference on a custom .pcd file
Performs semantic segmentation and saves the result with S3DIS colors
"""

import os
import sys
import time
import numpy as np
import torch
import open3d as o3d
from torch.nn.functional import softmax
import argparse

# Add module paths
sys.path.append('./third_party/KPConv-PyTorch')
sys.path.append('./third_party/sonata')

# KPConv imports
from utils.config import Config
from models.architectures import KPFCNN
from datasets.common import PointCloudDataset
from datasets.S3DIS import S3DISCustomBatch
import random

# Preprocessing imports
from pcd_preprocessor import load_and_preprocess_pcd, convert_to_kpconv_format


def parse_args():
    # --- Default Inference Parameters ---
    pcd_path = '/workspace/pcd_files/down0.01.pcd'
    model_path = '/workspace/kpconv_weights/Light_KPFCNN'
    output_dir = 'result_plys/kpconv_plys'
    downsampling_method = 'grid'
    voxel_size = 0.03
    chunk_size = 300000
    seed = 42
    device = 'cuda'
    # ------------------------------------

    parser = argparse.ArgumentParser(description="KPConv Semantic Segmentation Inference")
    parser.add_argument('--pcd_path', type=str, default=pcd_path,
                        help='Path to the input PCD file.')
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to the directory containing the trained KPConv model.')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='Directory to save the output PLY file.')
    parser.add_argument('--downsampling_method', type=str, default=downsampling_method, choices=['grid', 'voxel', 'random'],
                        help='Downsampling method.')
    parser.add_argument('--voxel_size', type=float, default=voxel_size,
                        help='Voxel size for downsampling.')
    parser.add_argument('--chunk_size', type=int, default=chunk_size,
                        help='Number of points to process in a single chunk.')
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default=device, choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
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
    
    print("✅ Random seed set for all libraries")


class CustomPointCloudDataset(PointCloudDataset):
    """Custom dataset for inferencing a single point cloud"""
    
    def __init__(self, config):
        # Initialize parent class without a dataset name
        super().__init__('Custom')
        self.config = config
        
        # S3DIS classes
        self.label_to_names = {
            0: 'ceiling',
            1: 'floor',
            2: 'wall',
            3: 'beam',
            4: 'column',
            5: 'window',
            6: 'door',
            7: 'chair',
            8: 'table',
            9: 'bookcase',
            10: 'sofa',
            11: 'board',
            12: 'clutter',
        }
        self.init_labels()
        self.ignored_labels = np.array([])


class KPConvInferencer:
    """Class for KPConv model inference"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initializer for the inferencer
        
        Args:
            model_path (str): Path to the folder with the trained model
            device (str): Device for computations ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = Config()
        self.config.load(model_path)
        
        print(f"Loaded model configuration: {self.config.dataset}")
        print(f"Number of classes: {self.config.num_classes}")
        print(f"Input feature dimensions: {self.config.in_features_dim}")
        
        # Create dataset to get class labels
        self.dataset = CustomPointCloudDataset(self.config)
        
        # Load model
        self._load_model()
        
        # S3DIS class colors (RGB in range 0-1)
        # self.class_colors = np.array([
        #     [0.7, 0.7, 0.7],   # 0: ceiling - серый
        #     [0.5, 0.3, 0.1],   # 1: floor - коричневый
        #     [0.8, 0.8, 0.6],   # 2: wall - бежевый
        #     [0.4, 0.2, 0.0],   # 3: beam - темно-коричневый
        #     [0.6, 0.6, 0.6],   # 4: column - темно-серый
        #     [0.0, 0.5, 0.8],   # 5: window - голубой
        #     [0.8, 0.4, 0.0],   # 6: door - оранжевый
        #     [0.0, 0.8, 0.0],   # 7: chair - зеленый
        #     [0.8, 0.0, 0.0],   # 8: table - красный
        #     [0.4, 0.0, 0.8],   # 9: bookcase - фиолетовый
        #     [0.8, 0.0, 0.8],   # 10: sofa - пурпурный
        #     [0.0, 0.0, 0.8],   # 11: board - синий
        #     [0.8, 0.8, 0.0],   # 12: clutter - желтый
        # ])
        # S3DIS class colors (RGB in range 0-1)
        self.class_colors = np.array([
            [82/255, 84/255, 163/255],    # 0: ceiling -> otherfurniture
            [152/255, 223/255, 138/255],    # 1: floor -> floor
            [174/255, 199/255, 232/255],    # 2: wall -> wall
            [82/255, 84/255, 163/255],    # 3: beam -> otherfurniture
            [82/255, 84/255, 163/255],    # 4: column -> otherfurniture
            [197/255, 176/255, 213/255],     # 5: window -> window
            [214/255, 39/255, 40/255],    # 6: door -> door
            [188/255, 189/255, 34/255],    # 7: chair -> chair
            [255/255, 152/255, 150/255],    # 8: table -> table
            [140/255, 86/255, 75/255],  # 9: sofa -> sofa
            [148/255, 103/255, 189/255],  # 10: bookcase -> bookshelf
            [196/255, 156/255, 148/255],  # 11: board -> picture
            [82/255, 84/255, 163/255],  # 12: clutter/other -> otherfurniture
        ])
        


    
    def _load_model(self):
        """Loads the trained model"""
        print("Loading model...")
        
        # Create model architecture
        self.model = KPFCNN(
            self.config, 
            self.dataset.label_values, 
            self.dataset.ignored_labels
        )
        
        # Load weights
        checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('chkp')]
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")
        
        # Take the latest checkpoint
        latest_checkpoint = sorted(checkpoints)[-1]
        chkp_path = os.path.join(checkpoint_path, latest_checkpoint)
        
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(chkp_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def _create_batch_from_points(self, points, features):
        """
        Creates a batch for a single point cloud
        
        Args:
            points (np.ndarray): Point coordinates (N, 3)
            features (np.ndarray): Point features (N, 5)
            
        Returns:
            batch: A batch object for the model
        """
        # Create dummy labels (not used in inference)
        labels = np.zeros(len(points), dtype=np.int64)
        stack_lengths = np.array([len(points)], dtype=np.int32)
        
        # Use the dataset method to create input data
        input_list = self.dataset.segmentation_inputs(
            points, features, labels, stack_lengths
        )
        
        # Add additional data for inference
        scales = np.array([1.0], dtype=np.float32)  # No scaling
        rots = np.eye(3, dtype=np.float32)[np.newaxis, :]  # No rotation
        cloud_inds = np.array([0], dtype=np.int32)  # Cloud index
        point_inds = np.array([0], dtype=np.int32)  # Point index
        input_inds = np.arange(len(points), dtype=np.int32)  # Indices of all points
        
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]
        
        # Create batch object
        batch = S3DISCustomBatch([input_list])
        
        return batch
    
    def predict(self, points, features, chunk_size=30000):
        """
        Performs prediction for a point cloud, splitting it into chunks
        
        Args:
            points (np.ndarray): Point coordinates (N, 3)
            features (np.ndarray): Point features (N, 5)
            chunk_size (int): Size of chunks for processing
            
        Returns:
            np.ndarray: Predicted classes for each point
        """
        print("Performing inference...")
        
        # Clear cache before starting inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   GPU cache cleared before inference")
        
        num_points = len(points)
        print(f"Total number of points: {num_points}")
        
        total_inference_time = 0.0
        
        # If there are too many points, split into chunks
        if num_points > chunk_size:
            print(f"Splitting into chunks of {chunk_size} points...")
            
            all_predictions = []
            all_probs = []
            
            for i in range(0, num_points, chunk_size):
                end_idx = min(i + chunk_size, num_points)
                chunk_points = points[i:end_idx]
                chunk_features = features[i:end_idx]
                
                print(f"Processing chunk {i//chunk_size + 1}: points {i}-{end_idx}")
                
                # Clear GPU cache before each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                chunk_pred, chunk_probs, chunk_time = self._predict_chunk(chunk_points, chunk_features)
                all_predictions.append(chunk_pred)
                all_probs.append(chunk_probs)
                total_inference_time += chunk_time
            
            predictions = np.concatenate(all_predictions)
            probs = np.concatenate(all_probs)
        else:
            predictions, probs, inference_time = self._predict_chunk(points, features)
            total_inference_time = inference_time
        
        print(f"Predictions received for {len(predictions)} points")
        print(f"Total inference time: {total_inference_time:.2f} seconds")
        
        return predictions, probs
    
    def _predict_chunk(self, points, features):
        """Prediction for a single data chunk"""
        # Create batch
        batch = self._create_batch_from_points(points, features)
        
        # Move to device
        if 'cuda' in str(self.device):
            batch.to(self.device)
        
        # Perform prediction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(batch, self.config)
            inference_time = time.time() - start_time
            
            print(f"  Chunk inference time: {inference_time:.2f} seconds")
        
        # Get class probabilities
        probs = softmax(outputs, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        
        return predictions, probs, inference_time
    
    def colorize_predictions(self, points, predictions):
        """
        Creates a colored point cloud based on predictions
        
        Args:
            points (np.ndarray): Point coordinates (N, 3)
            predictions (np.ndarray): Predicted classes (N,)
            
        Returns:
            o3d.geometry.PointCloud: Colored point cloud
        """
        # Create a point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Assign colors based on predictions
        colors = np.zeros((len(points), 3))
        for i, pred in enumerate(predictions):
            if pred < len(self.class_colors):
                colors[i] = self.class_colors[pred]
            else:
                colors[i] = [0.5, 0.5, 0.5]  # Gray for unknown classes
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd


def main():
    """Main function to demonstrate inference"""
    args = parse_args()
    
    set_random_seed(args.seed)

    print("=" * 80)
    print("KPCONV INFERENCE - SEMANTIC SEGMENTATION")
    print("=" * 80)
    
    # Create an informative output filename
    input_filename = os.path.splitext(os.path.basename(args.pcd_path))[0]
    model_name = os.path.basename(args.model_path)
    output_filename = (f"{input_filename}_KPConv_{model_name}_"
                      f"downsample_{args.downsampling_method}_voxel{args.voxel_size}m_"
                      f"chunk{args.chunk_size}_segmented_seed_{args.seed}.ply")
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Input file: {args.pcd_path}")
    print(f"Model: {model_name} (from {args.model_path})")
    print(f"Parameters: downsampling={args.downsampling_method}, voxel_size={args.voxel_size}m, chunk_size={args.chunk_size}, device={args.device}")
    print(f"Output file: {output_path}")
    
    try:
        # 0. Clear GPU cache before starting
        print("\n0. Clearing GPU cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   CUDA not available, using CPU")
        
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing PCD file...")
        point_dict = load_and_preprocess_pcd(
            file_path=args.pcd_path,
            downsampling_method=args.downsampling_method,
            voxel_size=args.voxel_size,
            add_segmentation=False
        )
        
        # 2. Convert to KPConv format
        print("\n2. Converting data to KPConv format...")
        points, features = convert_to_kpconv_format(point_dict)
        print(f"   Points shape: {points.shape}")
        print(f"   Features shape: {features.shape}")
        
        # 3. Create inferencer and load model
        print("\n3. Initializing model...")
        # Try CUDA, fall back to CPU if it fails
        try:
            # Try to use the device specified in the arguments
            inferencer = KPConvInferencer(args.model_path, device=args.device)
            print("\n4. Performing semantic segmentation...")
            predictions, probs = inferencer.predict(points, features, chunk_size=args.chunk_size)
        except RuntimeError as e:
            # If CUDA runs out of memory and 'cuda' was selected, switch to 'cpu'
            if "CUDA out of memory." in str(e) and args.device == 'cuda':
                print("   GPU memory insufficient, switching to CPU...")
                torch.cuda.empty_cache() # Clear cache before switching
                inferencer = KPConvInferencer(args.model_path, device='cpu')
                print("\n4. Performing semantic segmentation on CPU...")
                predictions, probs = inferencer.predict(points, features, chunk_size=args.chunk_size)
            else:
                raise e
        
        # 5. Create colored point cloud
        print("\n5. Creating colored point cloud...")
        colored_pcd = inferencer.colorize_predictions(points, predictions)
        
        # 6. Save the result
        print(f"\n6. Saving result to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(output_path, colored_pcd)
        
        print("\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"   Processed points: {len(points)}")
        print(f"   Result saved to: {output_path}")
        print(f"   Classes found: {len(np.unique(predictions))}")

        # Class statistics
        print("\nClass statistics:")
        unique, counts = np.unique(predictions, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = inferencer.dataset.label_to_names.get(class_id, f"unknown_{class_id}")
            percentage = count / len(predictions) * 100
            print(f"  {class_name}: {count} voxels ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
