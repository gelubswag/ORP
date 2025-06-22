from matplotlib import pyplot as plt
import numpy as np
from seaborn import scatterplot
from pandas import DataFrame

from dataset_parser import labels, data, features_list
from clusters import elbow_method, data_with_clusters, get_cluster_statistics, get_true_cluster_centers
from config import settings


def plot_elbow():
    """Построение графика метода локтя"""
    inertia_values = elbow_method()
    print("Значения инерции:", inertia_values)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, settings.MAX_CLUSTERS), inertia_values, 'bo-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Инерция')
    plt.title('Метод локтя для определения оптимального количества кластеров')
    plt.grid(True)
    plt.show()


def plot_data(data: DataFrame, labels: list):
    """Построение графиков кластеризации для различных количеств кластеров"""

    for n_clusters in range(2, settings.MAX_CLUSTERS + 1):  # Начинаем с 2 кластеров
        print(f"\nОбработка {n_clusters} кластеров...")

        data_clustered, centers = data_with_clusters(n_clusters)

        # Получаем истинные центры кластеров (вычисленные из данных)
        true_centers = get_true_cluster_centers(data_clustered, n_clusters)

        # Сравниваем центры
        print(f"\nСравнение центров для {n_clusters} кластеров:")
        print("Центры из KMeans (после обратной трансформации):")
        for i, center in enumerate(centers):
            print(f"  Кластер {i}: {dict(zip(features_list, center))}")

        print("Истинные центры (вычисленные из данных):")
        for i, center in enumerate(true_centers):
            print(f"  Кластер {i}: {dict(zip(features_list, center))}")

        # Используем истинные центры для визуализации
        centers_to_plot = true_centers

        # Сохраняем результаты кластеризации
        data_clustered.to_csv(
            path_or_buf=settings.CLUSTERS_DIR + f"{n_clusters}.csv",
            sep=",", index=False)
        data_clustered.to_excel(
            settings.CLUSTERS_DIR + f"{n_clusters}.xlsx",
            index=False)

        # Получаем и выводим статистику
        stats = get_cluster_statistics(data_clustered, n_clusters)
        print(f"\nСтатистика для {n_clusters} кластеров:")
        for cluster_name, cluster_stats in stats.items():
            print(f"\n{cluster_name}:")
            for feature, feature_stats in cluster_stats.items():
                print(
                    f"  {feature}: mean={feature_stats['mean']:.2f}, "
                    f"std={feature_stats['std']:.2f}, count={feature_stats['count']}"
                    )

        # Строим графики для всех пар фичей
        num_features = len(features_list)

        if num_features >= 2:
            # Создаем графики для всех пар фичей
            pair_count = 0
            for i in range(num_features):
                for j in range(num_features):
                    if i >= j:
                        continue

                    pair_count += 1
                    plt.figure(figsize=(12, 8))

                    feature_y = features_list[i]
                    feature_x = features_list[j]

                    # Строим scatter plot с цветовой кодировкой кластеров
                    scatter = scatterplot(
                        x=feature_x, y=feature_y,
                        data=data_clustered, hue='Cluster',
                        palette='tab10', s=80, alpha=0.8
                    )
                    # Добавляем истинные центры кластеров
                    for k, center in enumerate(centers_to_plot):
                        plt.scatter(
                            center[j], center[i],
                            c='red', marker='X', s=300,
                            edgecolors='black', linewidths=2,
                            label=f'Центр кластера {k}' if k == 0 else ""
                        )

                    plt.xlabel(feature_x, fontsize=12)
                    plt.ylabel(feature_y, fontsize=12)
                    plt.title(f'{n_clusters} кластеров: {feature_y} vs {feature_x}', fontsize=14)

                    # Улучшаем легенду
                    handles, labels_legend = plt.gca().get_legend_handles_labels()
                    if len(handles) > n_clusters:  # Если есть центры
                        cluster_handles = handles[:n_clusters]
                        cluster_labels = [f'Кластер {i}' for i in range(n_clusters)]
                        center_handle = handles[n_clusters]
                        center_label = 'Центры кластеров'

                        plt.legend(
                            cluster_handles + [center_handle],
                            cluster_labels + [center_label],
                            loc='best', fontsize=10
                            )
                    else:
                        plt.legend(loc='best', fontsize=10)

                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

            plt.show()

        elif num_features == 1:
            # Если только одна фича, строим гистограмму и точечный график
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Гистограмма
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            for cluster_id in range(n_clusters):
                cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
                ax1.hist(cluster_data[features_list[0]], alpha=0.7,
                        label=f'Кластер {cluster_id}', bins=15,
                        color=colors[cluster_id])

            ax1.set_xlabel(features_list[0])
            ax1.set_ylabel('Частота')
            ax1.set_title(f'{n_clusters} кластеров: Распределение {features_list[0]}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Точечный график с индексами
            for cluster_id in range(n_clusters):
                cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
                indices = cluster_data.index
                ax2.scatter(
                    indices, cluster_data[features_list[0]],
                    alpha=0.7, label=f'Кластер {cluster_id}',
                    color=colors[cluster_id], s=60
                    )

            # Добавляем центры
            for k, center in enumerate(centers_to_plot):
                ax2.axhline(
                    y=center[0], color='red', linestyle='--',
                    linewidth=2, alpha=0.8,
                    label='Центры кластеров' if k == 0 else ""
                    )

            ax2.set_xlabel('Индекс записи')
            ax2.set_ylabel(features_list[0])
            ax2.set_title(f'{n_clusters} кластеров: {features_list[0]} по индексам')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        # Дополнительно: матричный график для многомерных данных
        if num_features >= 3:
            plot_cluster_matrix(data_clustered, n_clusters, features_list, centers_to_plot)

        input(f"Нажмите Enter для продолжения ({n_clusters} кластеров обработано)...")


def plot_cluster_matrix(data_clustered, n_clusters, features_list, centers):
    """Построение матричного графика для многомерных данных"""
    num_features = len(features_list)
    fig, axes = plt.subplots(
        num_features, num_features,
        figsize=(4*num_features, 4*num_features)
        )

    if num_features == 1:
        axes = [[axes]]
    elif num_features == 2:
        axes = [axes] if axes.ndim == 1 else axes

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i][j] if num_features > 1 else axes[0][0]

            if i == j:
                # Диагональные элементы - гистограммы
                for cluster_id in range(n_clusters):
                    cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
                    ax.hist(
                        cluster_data[features_list[i]], alpha=0.6,
                        bins=15, color=colors[cluster_id],
                        label=f'Кластер {cluster_id}'
                        )

                ax.set_xlabel(features_list[i])
                ax.set_ylabel('Частота')
                ax.set_title(f'Распределение {features_list[i]}')

            else:
                # Недиагональные элементы - scatter plots
                for cluster_id in range(n_clusters):
                    cluster_data = data_clustered[data_clustered['Cluster'] == cluster_id]
                    ax.scatter(
                        cluster_data[features_list[j]],
                        cluster_data[features_list[i]],
                        alpha=0.7, color=colors[cluster_id],
                        label=f'Кластер {cluster_id}', s=40
                        )

                # Добавляем центры кластеров
                for k, center in enumerate(centers):
                    ax.scatter(
                        center[j], center[i],
                        c='red', marker='X', s=150,
                        edgecolors='black', linewidths=1
                        )

                ax.set_xlabel(features_list[j])
                ax.set_ylabel(features_list[i])

            ax.grid(True, alpha=0.3)

            # Добавляем легенду только в правый верхний угол
            if i == 0 and j == num_features - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(
        f'Матричный график кластеризации ({n_clusters} кластеров)',
        fontsize=16, y=0.98
        )
    plt.tight_layout()
    plt.show()


# Запуск анализа
if __name__ == "__main__":
    print("Начинаем анализ кластеризации...")
    print("Построение графика метода локтя...")
    plot_elbow()
    print("\nПостроение графиков кластеризации...")
    plot_data(data, labels)
