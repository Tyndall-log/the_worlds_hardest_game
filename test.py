from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt
import sys

class MyWidget(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("PyQt6 Key Event Test")
		self.setGeometry(100, 100, 400, 300)
		self.pressed_keys = set()
		self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # 키 입력을 받기 위한 포커스 설정

	def keyPressEvent(self, event):
		key = event.key()
		if key == Qt.Key.Key_W:
			self.pressed_keys.add('w')
		elif key == Qt.Key.Key_A:
			self.pressed_keys.add('a')
		elif key == Qt.Key.Key_S:
			self.pressed_keys.add('s')
		elif key == Qt.Key.Key_D:
			self.pressed_keys.add('d')
		print("Pressed:", self.pressed_keys)

	def keyReleaseEvent(self, event):
		key = event.key()
		if key == Qt.Key.Key_W:
			self.pressed_keys.discard('w')
		elif key == Qt.Key.Key_A:
			self.pressed_keys.discard('a')
		elif key == Qt.Key.Key_S:
			self.pressed_keys.discard('s')
		elif key == Qt.Key.Key_D:
			self.pressed_keys.discard('d')
		print("Released:", self.pressed_keys, key)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	w = MyWidget()
	w.show()
	sys.exit(app.exec())