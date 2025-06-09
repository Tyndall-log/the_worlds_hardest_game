import sys
import math
from PyQt6.QtWidgets import (
	QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QGridLayout,
	QTabWidget, QHBoxLayout, QRadioButton, QGroupBox, QScrollArea, QListWidget, QSizePolicy,
	QLineEdit, QLabel,
	QFileDialog,
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
		self.tabs.addTab(self.default_screen, "홈")

	def create_default_screen(self):
		widget = QWidget()
		main_layout = QGridLayout()
		layout1 = QHBoxLayout()
		layout2 = QHBoxLayout()

		play_button = QPushButton("플레이 탭 열기")
		play_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		play_button.clicked.connect(lambda: self.add_tab("플레이", self.create_play_tab()))

		map_editor_button = QPushButton("에디터 탭 열기")
		map_editor_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		map_editor_button.clicked.connect(lambda: self.add_tab("에디터", self.create_map_editor_tab()))

		replay_button = QPushButton("리플레이 탭 열기")
		replay_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		replay_button.clicked.connect(lambda: self.add_tab("리플레이", self.create_replay_tab()))

		main_layout.addWidget(play_button, 0, 0)
		main_layout.addWidget(map_editor_button, 0, 1)
		main_layout.addWidget(replay_button, 1, 0)
		widget.setLayout(main_layout)
		return widget

	def create_play_tab(self):
		widget = QWidget()
		main_layout = QVBoxLayout()

		mode_group = QGroupBox("모드 선택")
		mode_layout = QVBoxLayout()

		ai_train = QRadioButton("AI 학습 모드")
		ai_test = QRadioButton("AI 평가 모드")
		solo = QRadioButton("혼자 하기(맵 테스트용)")
		ai_train.setChecked(True)

		mode_layout.addWidget(ai_train)
		mode_layout.addWidget(ai_test)
		mode_layout.addWidget(solo)
		# mode_layout.addWidget(solo)
		mode_group.setLayout(mode_layout)

		control_layout = QHBoxLayout()
		start_button = QPushButton("시작")
		stop_button = QPushButton("중단")
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

	def create_replay_tab(self):
		widget = QWidget()
		layout = QVBoxLayout()
		widget.setLayout(layout)
		return widget

	def add_tab(self, name, content_widget):
		index = self.tabs.addTab(content_widget, name)
		self.tabs.setCurrentIndex(index)

	def close_tab(self, index):
		self.tabs.removeTab(index)
		if self.tabs.count() == 0:
			self.tabs.addTab(self.default_screen, "홈")


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec())
