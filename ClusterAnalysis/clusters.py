from sklearn.cluster import KMeans
import numpy as np

from dataset_parser import features_scaled, data, scaler, features_list
from config import settings


def elbow_method():
    """Метод локтя для определения оптимального количества кластеров"""
    inertia = []
    for i in range(1, settings.MAX_CLUSTERS):
        kmeans = KMeans(n_clusters=i, random_state=100)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
    return inertia


def data_with_clusters(n_clusters):
    """Кластеризация данных с корректным вычислением центров кластеров"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=100)
    kmeans.fit(features_scaled)

    # Получаем центры кластеров в масштабированном пространстве
    centers_scaled = kmeans.cluster_centers_

    # Обратное преобразование для получения центров в исходном масштабе
    centers_original = scaler.inverse_transform(centers_scaled)

    # Добавляем колонку кластеров к данным
    data_clustered = data.copy()
    data_clustered['Cluster'] = kmeans.fit_predict(features_scaled)

    print("Центры кластеров (в исходном масштабе):")
    for i, center in enumerate(centers_original):
        print(f"Кластер {i}: {dict(zip(features_list, center))}")

    # Дополнительная проверка: вычисляем средние значения для каждого кластера
    print("\nПроверка - средние значения кластеров из данных:")
    for cluster_id in range(n_clusters):
        cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
        cluster_means = {}
        for feature in features_list:
            cluster_means[feature] = cluster_data[feature].mean()
        print(f"Кластер {cluster_id}: {cluster_means}")

    return data_clustered, centers_original


def get_cluster_statistics(data_clustered, n_clusters):
    """Получить статистику по кластерам"""
    stats = {}
    for cluster_id in range(n_clusters):
        cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
        cluster_stats = {}

        for feature in features_list:
            cluster_stats[feature] = {
                'mean': cluster_data[feature].mean(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max(),
                'count': len(cluster_data)
            }

        stats[f'Cluster_{cluster_id}'] = cluster_stats
    return stats


def get_true_cluster_centers(data_clustered, n_clusters):
    """Вычисляет истинные центры кластеров на основе исходных данных"""
    true_centers = []

    for cluster_id in range(n_clusters):
        cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
        center = []

        for feature in features_list:
            center.append(cluster_data[feature].mean())

        true_centers.append(center)

    return np.array(true_centers)
