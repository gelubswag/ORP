import numpy as np
import os

# Типы данных для колонок
dtypes = {
    'number_courses': np.int32,
    'time_study': np.float32,
    'Marks': np.float32,
    'Sum of Females Life Expectancy': np.float32,
    'Sum of Life Expectancy (both sexes)': np.float32,
    'Sum of Males Life Expectancy': np.float32,
}

# Фичи для каждого датасета
features_by_dataset = {
    'Student_Marks.csv': ['number_courses', 'time_study', 'Marks'],
    'life_expectancy.csv': [
        'Sum of Females Life Expectancy',
        'Sum of Life Expectancy (both sexes)',
        'Sum of Males Life Expectancy'
    ]
}

# Веса (ценность) фичей для каждого датасета
feature_weights = {
    'Student_Marks.csv': {
        'number_courses': 0.2,
        'time_study': 0.3,
        'Marks': 0.5
    },
    'life_expectancy.csv': {
        'Sum of Females Life Expectancy': 0.4,
        'Sum of Life Expectancy (both sexes)': 0.3,
        'Sum of Males Life Expectancy': 0.3
    }
}


def get_features_for_dataset(dataset_path):
    """Получить список фичей для конкретного датасета"""
    dataset_name = os.path.basename(dataset_path)
    return features_by_dataset.get(dataset_name, [])


def get_weights_for_dataset(dataset_path):
    """Получить веса фичей для конкретного датасета"""
    dataset_name = os.path.basename(dataset_path)
    return feature_weights.get(dataset_name, {})


def get_features_indices(features):
    """Создать словарь индексов для фичей"""
    return {feature: idx for idx, feature in enumerate(features)}
