from sklearn.cluster import KMeans

from dataset_parser import features_scaled, data, scaler
from config import settings


def elbow_method():
    inertia = []
    for i in range(1, settings.MAX_CLUSTERS):
        kmeans = KMeans(n_clusters=i, random_state=100)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
    return inertia


def data_with_clusters(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=100)
    kmeans.fit(features_scaled)
    centers = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers)
    data['Cluster'] = kmeans.fit_predict(features_scaled)
    print(centers)
    return data, centers
