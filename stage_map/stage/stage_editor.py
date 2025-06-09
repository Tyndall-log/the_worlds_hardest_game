import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

from stage_map.env import Environment
from model.model3 import PPOModel


class ReinforcementLearningWorker(QThread):
	"""
	강화학습 알고리즘을 실행하는 작업 스레드
	"""
	update_signal = pyqtSignal(np.ndarray)  # NumPy 배열을 전달할 시그널

	def __init__(self):
		super().__init__()
		self.running = True  # 루프 제어 변수
		self.player_num = 16
		self.fps: int = 30
		self.env = Environment("", player_num=self.player_num, fps=self.fps)
		self.env.reset()
		self.obs = self.env.observation
		self.model = PPOModel(3, 9)

	def run(self):
		"""
		알고리즘 실행 루프
		"""
		model = self.model
		device = torch.device("mps")
		model.to(device)
		# model.load_state_dict(torch.load("model.pth"))
		model.eval()
		hidden_state = model.init_hidden_states(self.player_num)
		hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
		while self.running:
			env = self.env
			# 강화학습 알고리즘 실행
			x = torch.tensor(self.obs, dtype=torch.float32).permute(0, 3, 1, 2)
			x = x.to(device)
			policy, value, hidden_state = model(x, hidden_state)
			actions = policy.argmax(dim=1).cpu().numpy()

			# actions = np.random.randint(0, 9, size=self.player_num)  # 랜덤 행동
			obs, rewards, dones, infos = env.step(actions)
			self.obs = obs
			image = env.render()

			# 결과를 GUI로 전달
			self.update_signal.emit(image)

	def stop(self):
		"""
		알고리즘 중단
		"""
		self.running = False
		self.quit()
		self.wait()


class EnvironmentVisualizer(QMainWindow):
	def __init__(self, fps=60):
		super().__init__()

		# 기본 설정
		self.setWindowTitle("Environment Visualizer")
		dprf = self.devicePixelRatioF()
		self.window_w = int(math.ceil(1100 / dprf))
		self.window_h = int(math.ceil(800 / dprf))
		self.setGeometry(100, 100, self.window_w, self.window_h)

		# QGraphicsScene과 QGraphicsView 추가
		self.scene = QGraphicsScene()
		self.view = QGraphicsView(self.scene, self)
		self.view.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # QGraphicsView 포커스 비활성화
		self.setCentralWidget(self.view)

		# FPS 설정
		self.fps = fps
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_frame)
		self.timer.start(1000 // self.fps)  # FPS에 맞춰 타이머 설정

		# 이미지 데이터
		self.image_data = np.random.randint(0, 256, size=(800, 1100, 3), dtype=np.uint8)

		# 강화학습 작업 스레드
		self.worker = ReinforcementLearningWorker()
		self.worker.update_signal.connect(self.set_image_data)  # 스레드에서 데이터 수신
		self.worker.start()

	def update_frame(self):
		"""
		FPS에 따라 화면 업데이트
		"""

		# NumPy 배열을 QImage로 변환
		height, width, channels = self.image_data.shape
		bytes_per_line = channels * width
		q_image = QImage(self.image_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

		# QPixmap으로 변환 후 QGraphicsPixmapItem으로 렌더링
		pixmap = QPixmap.fromImage(q_image)
		pixmap.setDevicePixelRatio(self.devicePixelRatioF())
		self.scene.clear()
		self.scene.addPixmap(pixmap)

	def set_image_data(self, image_data):
		"""
		이미지 데이터 설정
		"""
		self.image_data = image_data

	def keyPressEvent(self, event):
		"""
		키보드 입력 처리
		"""
		print(f"Key pressed: {event.key()}")
		# esc
		if event.key() == Qt.Key.Key_Escape:
			self.close()

	def closeEvent(self, event):
		"""
		창 닫기 이벤트 처리
		"""
		self.timer.stop()  # 타이머 중지
		self.worker.stop()  # 작업 스레드 종료
		event.accept()  # 창 닫기 허용


if __name__ == "__main__":
	app = QApplication(sys.argv)
	fps = int(app.primaryScreen().refreshRate())
	visualizer = EnvironmentVisualizer(fps=60)
	visualizer.show()
	sys.exit(app.exec())
