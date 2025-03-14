from matplotlib import pyplot as plt
import numpy as np
from seaborn import scatterplot
from pandas import DataFrame

from dataset_parser import labels, data, scaler

from dataset_dtypes import dtypes, features_indices

from clusters import elbow_method, data_with_clusters

from config import settings

def plot_elbow():
    print(elbow_method())
    plt.plot(range(1, 11), elbow_method())
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def plot_data(data: DataFrame, labels: np.ndarray):
    for n_clusters in range(1, settings.MAX_CLUSTERS+1):
        data_clusters, centers = data_with_clusters(n_clusters)
        data_clusters.to_csv(
            path_or_buf=settings.CLUSTERS_DIR + f"{n_clusters}.csv",
            sep=",")
        data_clusters.to_excel(
            settings.CLUSTERS_DIR + f"{n_clusters}.xlsx",
            )
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i >= j:
                    continue
                plt.figure(n_clusters * 100 + i * 10 + j)
                plt.title(f'{n_clusters} Clusters, {labels[i]}: Y, {labels[j]}: X')
                scatterplot(x=labels[j], y=labels[i], data=data_clusters, hue='Cluster', palette='pastel')
                for center in centers:
                    if i in features_indices.keys():
                        if j in features_indices.keys():
                            plt.scatter(
                                center[features_indices[j]],
                                center[features_indices[i]],
                                c='red',
                                marker='x'
                                )
                        else:
                            plt.scatter(center[features_indices[j]] * np.ones(len(data_clusters[labels[i]])),
                                        data_clusters[labels[i]],
                                        c='red',
                                        marker='x'
                                        )
                    else:
                        if j in features_indices.keys():
                            plt.scatter(
                                center[features_indices[j]] * np.ones(
                                    len(data_clusters[labels[j]])
                                    ),
                                data_clusters[labels[i]],
                                c='red',
                                marker='x'
                                )
                        else:
                            plt.scatter(
                                data_clusters[labels[i]],
                                data_clusters[labels[j]],
                                c='red',
                                marker='x')

                plt.xlabel(labels[j])
                plt.ylabel(labels[i])
                plt.title(f'{labels[i]}: Y, {labels[j]}: X')
        plt.show()

        input(f"{n_clusters} Кластеров")


plot_elbow()
plot_data(data, labels)