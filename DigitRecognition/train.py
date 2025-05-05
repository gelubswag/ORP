import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def train_and_save_model():
    # Загрузка данных MNIST
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Создание модели CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Обучение
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Сохранение модели
    model.save('model/mnist_cnn.h5')
    print("Модель сохранена в model/mnist_cnn.h5")

if __name__ == "__main__":
    train_and_save_model()