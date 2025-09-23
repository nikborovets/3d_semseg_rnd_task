"""
Модуль для предобработки PCD файлов.
Содержит функции для downsampling и преобразования в формат словаря point.
"""

import gc
import numpy as np
import open3d as o3d


def downsample_point_cloud(pcd, method="voxel", voxel_size=0.025, target_points=100000):
    """
    Выполняет downsampling облака точек.
    
    Args:
        pcd (o3d.geometry.PointCloud): Исходное облако точек
        method (str): Метод downsampling ("voxel", "random" или "grid")
        voxel_size (float): Размер вокселя/сетки в метрах для voxel/grid downsampling
        target_points (int): Целевое количество точек для random downsampling
        
    Returns:
        o3d.geometry.PointCloud: Downsampled облако точек
    """
    print(f"Original points: {len(pcd.points)}")
    
    if method == "voxel":
        # Voxel downsampling (рекомендуется)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling (voxel_size={voxel_size}): {len(pcd_downsampled.points)}")
        
    elif method == "random":
        # Random downsampling (альтернатива)
        if len(pcd.points) > target_points:
            indices = np.random.choice(len(pcd.points), target_points, replace=False)
            pcd_downsampled = pcd.select_by_index(indices)
            print(f"After random downsampling (target={target_points}): {len(pcd_downsampled.points)}")
        else:
            pcd_downsampled = pcd
            print(f"No downsampling needed (points={len(pcd.points)} <= target={target_points})")
            
    elif method == "grid":
        coords = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        min_coords = coords.min(axis=0)
        quantized_coords = np.floor((coords - min_coords) / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(quantized_coords, axis=0, return_index=True)
        
        pcd_downsampled = o3d.geometry.PointCloud()
        pcd_downsampled.points = o3d.utility.Vector3dVector(coords[unique_indices])
        if colors is not None:
            pcd_downsampled.colors = o3d.utility.Vector3dVector(colors[unique_indices])
        if normals is not None:
            pcd_downsampled.normals = o3d.utility.Vector3dVector(normals[unique_indices])
            
        print(f"After grid downsampling (grid_size={voxel_size}): {len(pcd_downsampled.points)}")
        
    else:
        raise ValueError(f"Unknown downsampling method: {method}. Use 'voxel', 'random' or 'grid'")
    
    return pcd_downsampled


def pcd_to_point_dict(pcd, add_segmentation=False, segmentation_type="dummy"):
    """
    Преобразует Open3D PointCloud в формат словаря point.
    
    Args:
        pcd (o3d.geometry.PointCloud): Облако точек Open3D
        add_segmentation (bool): Добавлять ли сегментацию
        segmentation_type (str): Тип сегментации ("dummy", "zeros", "random")
        
    Returns:
        dict: Словарь в формате point с ключами coord, color, normal, segment20, segment200, instance
    """
    num_points = len(pcd.points)
    
    # Основные данные
    point = {
        "coord": np.array(pcd.points),
        "color": np.array(pcd.colors) if pcd.has_colors() else None,
        "normal": np.array(pcd.normals) if pcd.has_normals() else None,
    }
    
    # Обработка отсутствующих данных
    if point["color"] is None:
        print("Warning: No color data found, using default colors")
        point["color"] = np.ones((num_points, 3)) * 0.5  # Серый цвет по умолчанию
    
    if point["normal"] is None:
        print("Warning: No normal data found, computing normals")
        # Вычисляем нормали если их нет
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(point["coord"])
        pcd_temp.estimate_normals()
        point["normal"] = np.array(pcd_temp.normals)
        del pcd_temp
        gc.collect()
    
    # Добавление сегментации
    if add_segmentation:
        if segmentation_type == "dummy":
            # Фиктивная сегментация (все точки в одном классе)
            point["segment20"] = np.zeros(num_points, dtype=np.int8)
            point["segment200"] = np.zeros(num_points, dtype=np.int16)
            point["instance"] = np.zeros(num_points, dtype=np.int16)
            
        elif segmentation_type == "zeros":
            # Сегментация нулями (несегментированные точки)
            point["segment20"] = np.full(num_points, -1, dtype=np.int8)
            point["segment200"] = np.full(num_points, -1, dtype=np.int16)
            point["instance"] = np.full(num_points, -1, dtype=np.int16)
            
        elif segmentation_type == "random":
            # Случайная сегментация для тестирования
            point["segment20"] = np.random.randint(0, 20, num_points, dtype=np.int8)
            point["segment200"] = np.random.randint(0, 200, num_points, dtype=np.int16)
            point["instance"] = np.random.randint(0, 100, num_points, dtype=np.int16)
            
        else:
            raise ValueError(f"Unknown segmentation_type: {segmentation_type}")
    
    return point


def load_and_preprocess_pcd(file_path, 
                           downsampling_method="voxel", 
                           voxel_size=0.025, 
                           target_points=100000,
                           add_segmentation=True,
                           segmentation_type="dummy"):
    """
    Загружает PCD файл и выполняет полную предобработку.
    
    Args:
        file_path (str): Путь к PCD файлу
        downsampling_method (str): Метод downsampling ("voxel", "random" или "grid")
        voxel_size (float): Размер вокселя в метрах для voxel downsampling
        target_points (int): Целевое количество точек для random downsampling
        add_segmentation (bool): Добавлять ли сегментацию
        segmentation_type (str): Тип сегментации ("dummy", "zeros", "random")
        
    Returns:
        dict: Словарь в формате point с предобработанными данными
    """
    print(f"Loading PCD file: {file_path}")
    
    # Загружаем PCD файл
    pcd = o3d.io.read_point_cloud(file_path)
    
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud loaded from {file_path}")

    # Освобождаем память после загрузки
    gc.collect()
    print("Memory freed after PCD loading")
    
    # Выполняем downsampling
    pcd_downsampled = downsample_point_cloud(
        pcd, 
        method=downsampling_method, 
        voxel_size=voxel_size, 
        target_points=target_points
    )

    
    # Преобразуем в формат point
    point = pcd_to_point_dict(
        pcd_downsampled, 
        add_segmentation=add_segmentation, 
        segmentation_type=segmentation_type
    )
    
    print(f"Preprocessing completed. Final point count: {len(point['coord'])}")
    del pcd, pcd_downsampled  # Удаляем исходный PCD объект
    gc.collect()
    print("Memory freed after point dict conversion")
    
    return point


def convert_point_dict_format(point_dict, target_format="sonata"):
    """
    Конвертирует словарь point в нужный формат для различных моделей.
    
    Args:
        point_dict (dict): Исходный словарь point
        target_format (str): Целевой формат ("sonata", "scanet", "custom")
        
    Returns:
        dict: Словарь в целевом формате
    """
    if target_format == "sonata":
        # Формат для Sonata модели
        result = {
            "coord": point_dict["coord"],
            "color": point_dict["color"],
            "normal": point_dict["normal"]
        }
        
        # Добавляем сегментацию если есть
        if "segment20" in point_dict:
            result["segment20"] = point_dict["segment20"]
        if "segment200" in point_dict:
            result["segment200"] = point_dict["segment200"]
        if "instance" in point_dict:
            result["instance"] = point_dict["instance"]
            
        return result
        
    elif target_format == "scanet":
        # Формат для ScanNet (только одна сегментация)
        result = {
            "coord": point_dict["coord"],
            "color": point_dict["color"],
            "normal": point_dict["normal"]
        }
        
        # Используем segment20 как основную сегментацию
        if "segment20" in point_dict:
            result["segment"] = point_dict["segment20"]
        elif "segment200" in point_dict:
            result["segment"] = point_dict["segment200"]
        else:
            result["segment"] = np.zeros(len(point_dict["coord"]), dtype=np.int32)
            
        return result
        
    else:
        # Возвращаем как есть
        return point_dict.copy()


def get_preprocessing_summary(point_dict):
    """
    Получает краткую сводку по предобработанным данным.
    
    Args:
        point_dict (dict): Словарь point
        
    Returns:
        dict: Сводка по данным
    """
    summary = {
        "total_points": len(point_dict["coord"]),
        "has_color": point_dict["color"] is not None,
        "has_normal": point_dict["normal"] is not None,
        "has_segmentation": "segment20" in point_dict,
        "memory_mb": sum(arr.nbytes for arr in point_dict.values() if isinstance(arr, np.ndarray)) / (1024 * 1024)
    }
    
    if "coord" in point_dict:
        coord = point_dict["coord"]
        summary["coord_range"] = {
            "min": np.min(coord, axis=0).tolist(),
            "max": np.max(coord, axis=0).tolist(),
            "mean": np.mean(coord, axis=0).tolist()
        }
    
    return summary


def convert_to_kpconv_format(point_dict):
    """
    Конвертирует словарь point в формат, совместимый с KPConv модель.
    
    KPConv ожидает:
    - points: (N, 3) координаты xyz 
    - features: (N, 5) признаки [константа=1.0, R, G, B, normal_z]
    
    Args:
        point_dict (dict): Словарь point с ключами coord, color, normal
        
    Returns:
        tuple: (points, features) в формате для KPConv
            - points: np.ndarray (N, 3) координаты точек
            - features: np.ndarray (N, 5) признаки точек
    """
    if "coord" not in point_dict:
        raise ValueError("point_dict должен содержать ключ 'coord'")
    
    if "color" not in point_dict or point_dict["color"] is None:
        raise ValueError("point_dict должен содержать ключ 'color' с данными RGB")
        
    if "normal" not in point_dict or point_dict["normal"] is None:
        raise ValueError("point_dict должен содержать ключ 'normal' с данными нормалей")
    
    # Координаты точек (остаются как есть)
    points = point_dict["coord"].astype(np.float32)
    num_points = len(points)
    
    # Формируем признаки: [константа=1.0, R, G, B, normal_z]
    features = np.ones((num_points, 5), dtype=np.float32)
    
    # Столбцы 1,2,3: RGB цвета
    features[:, 1:4] = point_dict["color"].astype(np.float32)
    
    # Столбец 4: Z-компонента нормалей (вертикальная составляющая)
    features[:, 4] = point_dict["normal"][:, 2].astype(np.float32)
    
    return points, features


def convert_to_kpconv_format_extended(point_dict, feature_mode="rgb_normal_z"):
    """
    Расширенная версия конвертации в формат KPConv с различными режимами признаков.
    
    Args:
        point_dict (dict): Словарь point с данными
        feature_mode (str): Режим формирования признаков:
            - "rgb_normal_z": [1.0, R, G, B, normal_z] (по умолчанию)
            - "rgb_only": [1.0, R, G, B, 0.0] (только RGB)
            - "normal_only": [1.0, 0.0, 0.0, 0.0, normal_z] (только нормали)
            - "rgb_normal_all": [1.0, R, G, B, normal_magnitude] (RGB + величина нормали)
            
    Returns:
        tuple: (points, features) в формате для KPConv
    """
    if "coord" not in point_dict:
        raise ValueError("point_dict должен содержать ключ 'coord'")
    
    points = point_dict["coord"].astype(np.float32)
    num_points = len(points)
    features = np.ones((num_points, 5), dtype=np.float32)
    
    if feature_mode == "rgb_normal_z":
        # Стандартный режим: RGB + Z-компонента нормали
        if "color" in point_dict and point_dict["color"] is not None:
            features[:, 1:4] = point_dict["color"].astype(np.float32)
        if "normal" in point_dict and point_dict["normal"] is not None:
            features[:, 4] = point_dict["normal"][:, 2].astype(np.float32)
            
    elif feature_mode == "rgb_only":
        # Только RGB цвета
        if "color" in point_dict and point_dict["color"] is not None:
            features[:, 1:4] = point_dict["color"].astype(np.float32)
        features[:, 4] = 0.0  # Явно устанавливаем 0.0
        
    elif feature_mode == "normal_only":
        # Только нормали
        features[:, 1:4] = 0.0  # Явно устанавливаем RGB в 0.0
        if "normal" in point_dict and point_dict["normal"] is not None:
            features[:, 4] = point_dict["normal"][:, 2].astype(np.float32)
        
    elif feature_mode == "rgb_normal_all":
        # RGB + величина нормали
        if "color" in point_dict and point_dict["color"] is not None:
            features[:, 1:4] = point_dict["color"].astype(np.float32)
        if "normal" in point_dict and point_dict["normal"] is not None:
            # Вычисляем величину вектора нормали
            normal_magnitude = np.linalg.norm(point_dict["normal"], axis=1)
            features[:, 4] = normal_magnitude.astype(np.float32)
            
    else:
        raise ValueError(f"Неизвестный feature_mode: {feature_mode}")
    
    return points, features


def get_kpconv_format_info():
    """
    Возвращает информацию о формате данных для KPConv.
    
    Returns:
        dict: Информация о формате данных
    """
    return {
        "points_shape": "(N, 3)",
        "points_description": "xyz координаты точек в метрах",
        "features_shape": "(N, 5)", 
        "features_description": "Признаки: [константа=1.0, R, G, B, дополнительный_признак]",
        "features_columns": {
            0: "константа (всегда 1.0)",
            1: "Red компонента цвета [0-1]",
            2: "Green компонента цвета [0-1]", 
            3: "Blue компонента цвета [0-1]",
            4: "Z-компонента нормали или другой признак"
        },
        "data_types": {
            "points": "np.float32",
            "features": "np.float32"
        },
        "compatible_models": ["KPFCNN", "Light_KPFCNN"],
        "example_usage": """
# Базовое использование:
points, features = convert_to_kpconv_format(point_dict)

# Расширенное использование:
points, features = convert_to_kpconv_format_extended(point_dict, "rgb_normal_z")
        """
    }
