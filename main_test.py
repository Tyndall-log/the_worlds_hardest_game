import sys
import math
from PyQt6.QtWidgets import (
	QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
	QTabWidget, QHBoxLayout, QRadioButton, QGroupBox, QLabel, QScrollArea, QListWidget
)


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("세가게 AI")
		dprf = self.devicePixelRatioF()
		self.window_w = int(math.ceil(1280 / dprf))
		self.window_h = int(math.ceil(720 / dprf))
		self.setGeometry(100, 100, self.window_w, self.window_h)

		self.tabs = QTabWidget()
		self.tabs.setTabsClosable(True)
		self.tabs.tabCloseRequested.connect(self.close_tab)
		self.setCentralWidget(self.tabs)

		self.default_screen = self.create_default_screen()
		self.tabs.addTab(self.default_screen, "Home")

	def create_default_screen(self):
		widget = QWidget()
		layout = QHBoxLayout()

		play_button = QPushButton("Play")
		play_button.clicked.connect(lambda: self.add_tab("Play", self.create_play_tab()))

		map_editor_button = QPushButton("Map Editor")
		map_editor_button.clicked.connect(lambda: self.add_tab("Map Editor", self.create_map_editor_tab()))

		layout.addWidget(play_button)
		layout.addWidget(map_editor_button)
		widget.setLayout(layout)
		return widget

	def create_play_tab(self):
		widget = QWidget()
		main_layout = QVBoxLayout()

		mode_group = QGroupBox("Mode Selection")
		mode_layout = QVBoxLayout()

		ai_train = QRadioButton("AI Training Mode")
		ai_test = QRadioButton("AI Test Mode")
		solo = QRadioButton("Solo Mode")
		ai_train.setChecked(True)

		mode_layout.addWidget(ai_train)
		mode_layout.addWidget(ai_test)
		mode_layout.addWidget(solo)
		mode_group.setLayout(mode_layout)

		control_layout = QHBoxLayout()
		start_button = QPushButton("Start")
		stop_button = QPushButton("Stop")
		control_layout.addWidget(start_button)
		control_layout.addWidget(stop_button)

		main_layout.addWidget(mode_group)
		main_layout.addWidget(QLabel("Map Selection"))
		main_layout.addWidget(QPushButton("Select Map"))
		main_layout.addLayout(control_layout)

		widget.setLayout(main_layout)
		return widget

	def create_map_editor_tab(self):
		widget = QWidget()
		layout = QHBoxLayout()

		map_list = QListWidget()
		map_list.addItems(["Map 1", "Map 2", "Map 3"])

		map_view = QLabel("[ Map View ]")
		object_details = QLabel("[ Object Details ]")

		layout.addWidget(map_list)
		layout.addWidget(map_view)
		layout.addWidget(object_details)
		widget.setLayout(layout)
		return widget

	def add_tab(self, name, content_widget):
		index = self.tabs.addTab(content_widget, name)
		self.tabs.setCurrentIndex(index)

	def close_tab(self, index):
		self.tabs.removeTab(index)
		if self.tabs.count() == 0:
			self.tabs.addTab(self.default_screen, "Home")


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec())
