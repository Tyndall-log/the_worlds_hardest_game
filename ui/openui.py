from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from PyQt6 import uic
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)  # 직접 로드
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())