import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, 
    QHBoxLayout, QWidget, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint
import tensorflow as tf

class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.last_point = QPoint()
        self.pixmap = QPixmap(280, 280)
        self.pixmap.fill(Qt.GlobalColor.black)
        self.setPixmap(self.pixmap)

    def mouseMoveEvent(self, event):
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(Qt.GlobalColor.white, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(self.last_point, event.pos())
        painter.end()
        self.setPixmap(self.pixmap)
        self.last_point = event.pos()

    def mousePressEvent(self, event):
        self.last_point = event.pos()

    def clear(self):
        self.pixmap.fill(Qt.GlobalColor.black)
        self.setPixmap(self.pixmap)

class DigitRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Рисуйте цифру (MNIST)")
        self.setFixedSize(400, 500)
        
        # Загрузка модели
        self.model = tf.keras.models.load_model("model/mnist_cnn.h5")
        
        # Виджеты
        self.canvas = DrawingCanvas()
        self.result_label = QLabel("Результат: ")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 20px;")
        
        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.canvas.clear)
        
        self.recognize_button = QPushButton("Распознать")
        self.recognize_button.clicked.connect(self.recognize_digit)
        
        # Разметка
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.recognize_button)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(buttons_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def recognize_digit(self):
        # Преобразуем QPixmap в массив numpy
        qimage = self.canvas.pixmap.toImage().scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio)
        byte_str = qimage.bits().asstring(28 * 28 * 4)
        img_array = np.frombuffer(byte_str, dtype=np.uint8).reshape(28, 28, 4)
        img_gray = img_array[:, :, 0]  # Берём только канал R (для ч/б)
        img_gray = img_gray.astype('float32') / 255.0
        img_gray = img_gray.reshape(1, 28, 28, 1)
        
        # Предсказание
        prediction = self.model.predict(img_gray)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        self.result_label.setText(f"Результат: {predicted_digit} (точность: {confidence:.2%})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognizerApp()
    window.show()
    sys.exit(app.exec())