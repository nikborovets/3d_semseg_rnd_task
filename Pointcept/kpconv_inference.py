#!/usr/bin/env python3
"""
Скрипт для инференса KPConv модели на кастомном .pcd файле
Выполняет семантическую сегментацию и сохраняет результат с цветами S3DIS
"""

import os
import sys
import time
import numpy as np
import torch
import open3d as o3d
from torch.nn.functional import softmax
import argparse

# Добавляем пути к модулям
sys.path.append('./third_party/KPConv-PyTorch')
sys.path.append('./third_party/sonata')

# Импорты KPConv
from utils.config import Config
from models.architectures import KPFCNN
from datasets.common import PointCloudDataset
from datasets.S3DIS import S3DISCustomBatch
import random

# Импорты предобработки
from pcd_preprocessor import load_and_preprocess_pcd, convert_to_kpconv_format


def parse_args():
    parser = argparse.ArgumentParser(description="KPConv Semantic Segmentation Inference")
    parser.add_argument('--pcd_path', type=str, default='/workspace/pcd_files/down0.01.pcd',
                        help='Path to the input PCD file.')
    parser.add_argument('--model_path', type=str, default='/workspace/kpconv_weights/Light_KPFCNN',
                        help='Path to the directory containing the trained KPConv model.')
    parser.add_argument('--output_dir', type=str, default='result_plys/kpconv_plys',
                        help='Directory to save the output PLY file.')
    parser.add_argument('--downsampling_method', type=str, default='grid', choices=['grid', 'voxel', 'random'],
                        help='Downsampling method.')
    parser.add_argument('--voxel_size', type=float, default=0.03,
                        help='Voxel size for downsampling.')
    parser.add_argument('--chunk_size', type=int, default=300000,
                        help='Number of points to process in a single chunk.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference.')
    return parser.parse_args()


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
    
    print("✅ Random seed установлен для всех библиотек")


class CustomPointCloudDataset(PointCloudDataset):
    """Кастомный датасет для инференса одного облака точек"""
    
    def __init__(self, config):
        # Инициализируем родительский класс без имени датасета
        super().__init__('Custom')
        self.config = config
        
        # S3DIS классы
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
    """Класс для инференса KPConv модели"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Инициализация инференсера
        
        Args:
            model_path (str): Путь к папке с обученной моделью
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Используем устройство: {self.device}")
        
        # Загружаем конфигурацию
        self.config = Config()
        self.config.load(model_path)
        
        print(f"Загружена конфигурация модели: {self.config.dataset}")
        print(f"Количество классов: {self.config.num_classes}")
        print(f"Размерность признаков: {self.config.in_features_dim}")
        
        # Создаем датасет для получения меток классов
        self.dataset = CustomPointCloudDataset(self.config)
        
        # Загружаем модель
        self._load_model()
        
        # # Цвета для классов S3DIS (RGB в диапазоне 0-1)
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
        """Загружает обученную модель"""
        print("Загружаем модель...")
        
        # Создаем архитектуру модели
        self.model = KPFCNN(
            self.config, 
            self.dataset.label_values, 
            self.dataset.ignored_labels
        )
        
        # Загружаем веса
        checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('chkp')]
        
        if not checkpoints:
            raise ValueError(f"Не найдены чекпоинты в {checkpoint_path}")
        
        # Берем последний чекпоинт
        latest_checkpoint = sorted(checkpoints)[-1]
        chkp_path = os.path.join(checkpoint_path, latest_checkpoint)
        
        print(f"Загружаем чекпоинт: {latest_checkpoint}")
        
        checkpoint = torch.load(chkp_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Модель успешно загружена!")
    
    def _create_batch_from_points(self, points, features):
        """
        Создает batch для одного облака точек
        
        Args:
            points (np.ndarray): Координаты точек (N, 3)
            features (np.ndarray): Признаки точек (N, 5)
            
        Returns:
            batch: Объект batch для модели
        """
        # Создаем фиктивные метки (не используются при инференсе)
        labels = np.zeros(len(points), dtype=np.int64)
        stack_lengths = np.array([len(points)], dtype=np.int32)
        
        # Используем метод из датасета для создания входных данных
        input_list = self.dataset.segmentation_inputs(
            points, features, labels, stack_lengths
        )
        
        # Добавляем дополнительные данные для инференса
        scales = np.array([1.0], dtype=np.float32)  # Без масштабирования
        rots = np.eye(3, dtype=np.float32)[np.newaxis, :]  # Без поворота
        cloud_inds = np.array([0], dtype=np.int32)  # Индекс облака
        point_inds = np.array([0], dtype=np.int32)  # Индекс точки
        input_inds = np.arange(len(points), dtype=np.int32)  # Индексы всех точек
        
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]
        
        # Создаем batch объект
        batch = S3DISCustomBatch([input_list])
        
        return batch
    
    def predict(self, points, features, chunk_size=30000):
        """
        Выполняет предсказание для облака точек с разбивкой на части
        
        Args:
            points (np.ndarray): Координаты точек (N, 3)
            features (np.ndarray): Признаки точек (N, 5)
            chunk_size (int): Размер части для обработки
            
        Returns:
            np.ndarray: Предсказанные классы для каждой точки
        """
        print("Выполняем инференс...")
        
        # Очищаем кэш перед началом инференса
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   GPU кэш очищен перед инференсом")
        
        num_points = len(points)
        print(f"Общее количество точек: {num_points}")
        
        total_inference_time = 0.0
        
        # Если точек слишком много, разбиваем на части
        if num_points > chunk_size:
            print(f"Разбиваем на части по {chunk_size} точек...")
            
            all_predictions = []
            all_probs = []
            
            for i in range(0, num_points, chunk_size):
                end_idx = min(i + chunk_size, num_points)
                chunk_points = points[i:end_idx]
                chunk_features = features[i:end_idx]
                
                print(f"Обрабатываем часть {i//chunk_size + 1}: точки {i}-{end_idx}")
                
                # Очищаем кэш GPU перед каждой частью
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
        
        print(f"Предсказания получены для {len(predictions)} точек")
        print(f"Общее время инференса: {total_inference_time:.2f} секунд")
        
        return predictions, probs
    
    def _predict_chunk(self, points, features):
        """Предсказание для одной части данных"""
        # Создаем batch
        batch = self._create_batch_from_points(points, features)
        
        # Переносим на устройство
        if 'cuda' in str(self.device):
            batch.to(self.device)
        
        # Выполняем предсказание
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(batch, self.config)
            inference_time = time.time() - start_time
            
            print(f"  Время инференса части: {inference_time:.2f} секунд")
        
        # Получаем вероятности классов
        probs = softmax(outputs, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        
        return predictions, probs, inference_time
    
    def colorize_predictions(self, points, predictions):
        """
        Создает цветное облако точек на основе предсказаний
        
        Args:
            points (np.ndarray): Координаты точек (N, 3)
            predictions (np.ndarray): Предсказанные классы (N,)
            
        Returns:
            o3d.geometry.PointCloud: Цветное облако точек
        """
        # Создаем облако точек
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Назначаем цвета на основе предсказаний
        colors = np.zeros((len(points), 3))
        for i, pred in enumerate(predictions):
            if pred < len(self.class_colors):
                colors[i] = self.class_colors[pred]
            else:
                colors[i] = [0.5, 0.5, 0.5]  # Серый для неизвестных классов
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd


def main():
    """Основная функция для демонстрации инференса"""
    args = parse_args()
    
    set_random_seed(args.seed)

    print("=" * 80)
    print("KPCONV INFERENCE - СЕМАНТИЧЕСКАЯ СЕГМЕНТАЦИЯ")
    print("=" * 80)
    
    # Создаем информативное имя выходного файла
    input_filename = os.path.splitext(os.path.basename(args.pcd_path))[0]
    model_name = os.path.basename(args.model_path)
    output_filename = (f"{input_filename}_KPConv_{model_name}_"
                      f"downsample_{args.downsampling_method}_voxel{args.voxel_size}m_"
                      f"chunk{args.chunk_size}_segmented_seed_{args.seed}.ply")
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Входной файл: {args.pcd_path}")
    print(f"Модель: {model_name} (из {args.model_path})")
    print(f"Параметры: downsampling={args.downsampling_method}, voxel_size={args.voxel_size}m, chunk_size={args.chunk_size}, device={args.device}")
    print(f"Выходной файл: {output_path}")
    
    try:
        # 0. Очищаем GPU кэш перед началом
        print("\n0. Очищаем GPU кэш...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   GPU память очищена. Доступно: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   CUDA недоступна, используем CPU")
        
        # 1. Загружаем и предобрабатываем данные
        print("\n1. Загружаем и предобрабатываем PCD файл...")
        point_dict = load_and_preprocess_pcd(
            file_path=args.pcd_path,
            downsampling_method=args.downsampling_method,
            voxel_size=args.voxel_size,
            add_segmentation=False
        )
        
        # 2. Конвертируем в формат KPConv
        print("\n2. Конвертируем данные в формат KPConv...")
        points, features = convert_to_kpconv_format(point_dict)
        print(f"   Points shape: {points.shape}")
        print(f"   Features shape: {features.shape}")
        
        # 3. Создаем инференсер и загружаем модель
        print("\n3. Инициализируем модель...")
        # Пробуем CUDA, если не получается - используем CPU
        try:
            # Пытаемся использовать устройство, указанное в аргументах
            inferencer = KPConvInferencer(args.model_path, device=args.device)
            print("\n4. Выполняем семантическую сегментацию...")
            predictions, probs = inferencer.predict(points, features, chunk_size=args.chunk_size)
        except RuntimeError as e:
            # Если на CUDA не хватило памяти и был выбран 'cuda', переключаемся на 'cpu'
            if "CUDA out of memory." in str(e) and args.device == 'cuda':
                print("   GPU память недостаточна, переключаемся на CPU...")
                torch.cuda.empty_cache() # Очищаем кэш перед переключением
                inferencer = KPConvInferencer(args.model_path, device='cpu')
                print("\n4. Выполняем семантическую сегментацию на CPU...")
                predictions, probs = inferencer.predict(points, features, chunk_size=args.chunk_size)
            else:
                raise e
        
        # 5. Создаем цветное облако точек
        print("\n5. Создаем цветное облако точек...")
        colored_pcd = inferencer.colorize_predictions(points, predictions)
        
        # 6. Сохраняем результат
        print(f"\n6. Сохраняем результат в {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(output_path, colored_pcd)
        
        print("\n✅ ИНФЕРЕНС ЗАВЕРШЕН УСПЕШНО!")
        print(f"   Обработано точек: {len(points)}")
        print(f"   Результат сохранен: {output_path}")
        print(f"   Найдено классов: {len(np.unique(predictions))}")

        # Статистика по классам
        print("\nСтатистика классов:")
        unique, counts = np.unique(predictions, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = inferencer.dataset.label_to_names.get(class_id, f"unknown_{class_id}")
            percentage = count / len(predictions) * 100
            print(f"  {class_name}: {count} voxels ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
