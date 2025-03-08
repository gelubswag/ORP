from matplotlib import pyplot as plt
import numpy as np

from dataset_parser import labels, data


def plot_data(data: np.ndarray, labels: np.ndarray):
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i <= j:
                break

            plt.figure(i * 10 + j)
            plt.scatter(data[:, j], data[:, i])
            plt.grid()
            plt.xlabel(labels[j])
            plt.ylabel(labels[i])
            plt.title(f'{labels[i]}: Y, {labels[j]}: X')

    plt.show()


plot_data(data, labels)
