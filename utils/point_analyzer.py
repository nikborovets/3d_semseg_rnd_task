"""
Модуль для детального анализа данных Point Cloud.
Содержит функции для статистического анализа NumPy массивов в словаре point.
"""

import numpy as np
from utils.pcd_preprocessor import load_and_preprocess_pcd


def analyze_numpy_array(key, array):
    """
    Функция для детального анализа NumPy массива.

    Args:
        key (str): Название ключа в словаре
        array (np.ndarray): NumPy массив для анализа
    """
    print(f"\n{'=' * 60}")
    print(f"АНАЛИЗ КЛЮЧА: '{key}'")
    print(f"{'=' * 60}")

    # Основная информация о массиве
    print(f"Тип данных: {type(array)}")
    print(f"Форма массива: {array.shape}")
    print(f"Размерность: {array.ndim}")
    print(f"Общее количество элементов: {array.size}")
    print(f"Тип элементов: {array.dtype}")

    # Память
    memory_mb = array.nbytes / (1024 * 1024)
    print(f"Память (MB): {memory_mb:.2f}")

    # Статистические характеристики
    if array.size > 0:
        print("\nСТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:")
        print(f"Минимум: {np.min(array)}")
        print(f"Максимум: {np.max(array)}")
        print(f"Среднее: {np.mean(array):.6f}")
        print(f"Стандартное отклонение: {np.std(array):.6f}")
        print(f"Медиана: {np.median(array):.6f}")

        # Для целочисленных типов - уникальные значения
        if np.issubdtype(array.dtype, np.integer):
            unique_vals = np.unique(array)
            print(f"Количество уникальных значений: {len(unique_vals)}")
            if len(unique_vals) <= 20:  # Показываем только если не слишком много
                print(f"Уникальные значения: {unique_vals}")
            else:
                print(f"Первые 10 уникальных значений: {unique_vals[:10]}")
                print(f"Последние 10 уникальных значений: {unique_vals[-10:]}")

        # Для массивов с координатами/цветами/нормалями
        if key in ["coord", "color", "normal"] and array.ndim == 2:
            print("\nАНАЛИЗ ПО ОСЯМ:")
            for i in range(array.shape[1]):
                axis_data = array[:, i]
                print(
                    f"  Ось {i}: min={np.min(axis_data):.6f}, max={np.max(axis_data):.6f}, mean={np.mean(axis_data):.6f}"
                )

    # Первые несколько элементов для понимания структуры
    print("\nПЕРВЫЕ 5 ЭЛЕМЕНТОВ:")
    if array.ndim == 1:
        print(f"  {array[:5]}")
    elif array.ndim == 2:
        print(f"  {array[:5]}")
    else:
        print(f"  {array.flat[:5]}")


def analyze_point_dict(point_dict):
    """
    Функция для полного анализа словаря point с данными Point Cloud.

    Args:
        point_dict (dict): Словарь с данными Point Cloud
    """
    print(f"\n{'#' * 80}")
    print("ДЕТАЛЬНЫЙ АНАЛИЗ СЛОВАРЯ POINT")
    print(f"{'#' * 80}")

    for key in point_dict.keys():
        if isinstance(point_dict[key], np.ndarray):
            analyze_numpy_array(key, point_dict[key])
        else:
            print(f"\n{'=' * 60}")
            print(f"КЛЮЧ: '{key}' - НЕ NUMPY МАССИВ")
            print(f"Тип: {type(point_dict[key])}")
            print(f"Значение: {point_dict[key]}")
            print(f"{'=' * 60}")

    print(f"\n{'#' * 80}")
    print("ОБЩАЯ СВОДКА")
    print(f"{'#' * 80}")
    total_memory = sum(
        arr.nbytes for arr in point_dict.values() if isinstance(arr, np.ndarray)
    ) / (1024 * 1024)
    print(f"Общий объем данных: {total_memory:.2f} MB")
    print(
        f"Количество точек: {len(point_dict['coord']) if 'coord' in point_dict else 'N/A'}"
    )
    print(f"{'#' * 80}")


def get_point_summary(point_dict):
    """
    Получить краткую сводку по словарю point без подробного вывода.

    Args:
        point_dict (dict): Словарь с данными Point Cloud

    Returns:
        dict: Словарь с краткой статистикой
    """
    summary = {}

    for key, value in point_dict.items():
        if isinstance(value, np.ndarray):
            summary[key] = {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "size": value.size,
                "memory_mb": value.nbytes / (1024 * 1024),
                "min": float(np.min(value)) if value.size > 0 else None,
                "max": float(np.max(value)) if value.size > 0 else None,
                "mean": float(np.mean(value)) if value.size > 0 else None,
                "unique_count": len(np.unique(value))
                if np.issubdtype(value.dtype, np.integer)
                else None,
            }
        else:
            summary[key] = {"type": str(type(value)), "value": str(value)}

    return summary


if __name__ == "__main__":
    pcd_file_path = "/workspace/pcd_files/down0.01.pcd"
    point = load_and_preprocess_pcd(
        file_path=pcd_file_path,
        downsampling_method="voxel",
        voxel_size=0.05,
        # target_points=193982,
        add_segmentation=False,
    )
    analyze_point_dict(point)
