#!/usr/bin/env python3
"""
Главный модуль для анализа кластеризации данных с поддержкой весов фичей
"""

import os
from plots import plot_elbow, plot_data
from dataset_parser import data, labels, features_list, weights_dict, dataset_file
from clusters import data_with_clusters, get_cluster_statistics
from config import settings


def create_output_directory():
    """Создание директории для выходных файлов"""
    if not os.path.exists(settings.CLUSTERS_DIR):
        os.makedirs(settings.CLUSTERS_DIR)
        print(f"Создана директория: {settings.CLUSTERS_DIR}")


def print_dataset_info():
    """Вывод информации о загруженном датасете"""
    print("=" * 50)
    print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("=" * 50)
    print(f"Файл: {dataset_file}")
    print(f"Размер: {data.shape[0]} строк, {data.shape[1]} колонок")
    print(f"Используемые фичи: {features_list}")
    print(f"Веса фичей: {weights_dict}")
    print("=" * 50)


def interactive_clustering():
    """Интерактивная кластеризация с выбором количества кластеров"""
    while True:
        try:
            n_clusters = int(input(f"\nВведите количество кластеров (1-{settings.MAX_CLUSTERS-1}) или 0 для выхода: "))

            if n_clusters == 0:
                print("Выход из программы.")
                break

            if n_clusters < 1 or n_clusters >= settings.MAX_CLUSTERS:
                print(f"Количество кластеров должно быть от 1 до {settings.MAX_CLUSTERS-1}")
                continue

            print(f"\nВыполняется кластеризация на {n_clusters} кластеров...")

            # Получаем данные с кластерами
            data_clustered, centers = data_with_clusters(n_clusters)

            # Получаем статистику
            stats = get_cluster_statistics(data_clustered, n_clusters)

            # Выводим детальную статистику
            print(f"\nДетальная статистика для {n_clusters} кластеров:")
            for cluster_name, cluster_stats in stats.items():
                print(f"\n{cluster_name} (размер: {list(cluster_stats.values())[0]['count']} элементов):")
                for feature, feature_stats in cluster_stats.items():
                    print(f"  {feature}:")
                    print(f"    Среднее: {feature_stats['mean']:.3f}")
                    print(f"    Стд. отклонение: {feature_stats['std']:.3f}")
                    print(f"    Диапазон: [{feature_stats['min']:.3f}, {feature_stats['max']:.3f}]")

            # Сохраняем результаты
            output_file = os.path.join(settings.CLUSTERS_DIR, f"{n_clusters}_clusters.csv")
            data_clustered.to_csv(output_file, sep=",", index=False)
            print(f"\nРезультаты сохранены в: {output_file}")

        except ValueError:
            print("Пожалуйста, введите корректное число.")
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем.")
            break


def main():
    """Главная функция"""
    print("Добро пожаловать в систему анализа кластеризации!")

    try:
        # Создаем выходную директорию
        create_output_directory()

        # Выводим информацию о датасете
        print_dataset_info()

        # Выбор режима работы
        print("\nВыберите режим работы:")
        print("1. Автоматический анализ (метод локтя + все кластеры)")
        print("2. Интерактивный режим (выбор количества кластеров)")
        print("3. Только метод локтя")

        while True:
            try:
                choice = int(input("Ваш выбор (1-3): "))
                if choice in [1, 2, 3]:
                    break
                else:
                    print("Выберите 1, 2 или 3")
            except ValueError:
                print("Введите число от 1 до 3")

        if choice == 1:
            print("\nЗапуск автоматического анализа...")
            plot_elbow()
            plot_data(data, labels)
        elif choice == 2:
            print("\nЗапуск интерактивного режима...")
            plot_elbow()
            interactive_clustering()
        elif choice == 3:
            print("\nПостроение графика метода локтя...")
            plot_elbow()

    except Exception as e:
        print(f"Ошибка выполнения программы: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
