from glob import glob

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import settings
from dataset_dtypes import dtypes, get_features_for_dataset, get_weights_for_dataset


def parse_csv(
    file_mask: str = settings.DATASET_MASK,
            ) -> tuple:
    csv_files = glob(file_mask)

    if not csv_files:
        raise FileNotFoundError(f"Не найдены датасеты с маской: {file_mask}")

    if len(csv_files) > 1:
        print("Найдено несколько датасетов: ")

        for file_i in range(len(csv_files)):
            print(f"{file_i}: {csv_files[file_i]}")

        while True:
            try:
                choice = int(input("Выберите номер файла: "))
                file = csv_files[choice]
                break
            except (ValueError, IndexError) as e:
                print(f"Неверный номер файла: {e}")
    else:
        file = csv_files[0]

    print(f"Выбран датасет: {file}")

    # Получаем фичи и веса для выбранного датасета
    features_list = get_features_for_dataset(file)
    weights_dict = get_weights_for_dataset(file)

    if not features_list:
        raise ValueError(f"Не найдены фичи для датасета: {file}")

    df = pd.read_csv(file, delimiter=settings.CSV_DELIMITER, dtype=dtypes)

    # Проверяем наличие всех необходимых колонок
    missing_features = [f for f in features_list if f not in df.columns]
    if missing_features:
        raise ValueError(f"Отсутствуют колонки в датасете: {missing_features}")

    features_df = df[features_list]

    return (
        df,
        features_df, df.columns.tolist(),
        features_list,
        weights_dict,
        file
    )


def normalize_with_weights(features_df, weights_dict):
    """Нормализация фичей с учетом их весов"""
    # Сначала стандартизируем данные
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    if weights_dict:
        weights_array = np.array(
            [weights_dict.get(col, 1.0) for col in features_df.columns]
            )
        print(
            f"Применяемые веса к стандартизированным данным: {weights_array}"
            )

        # Применяем веса к стандартизированным фичам
        # Это изменит масштаб фичей относительно друг друга
        features_weighted = features_scaled * np.sqrt(weights_array)

        print("Веса применены к нормализованным данным")
    else:
        features_weighted = features_scaled
        print("Веса не найдены, используются равные веса для всех фичей")

    return features_weighted, scaler, weights_dict


# Парсим датасет
data, features_unscaled, labels, features_list, weights_dict, dataset_file = parse_csv(settings.DATASET_MASK)
features_scaled, scaler, weights = normalize_with_weights(features_unscaled, weights_dict)

print(f"Используемые фичи: {features_list}")
print(f"Веса фичей: {weights_dict}")
print(f"Форма данных: {data.shape}")
print(f"Форма нормализованных фичей: {features_scaled.shape}")
