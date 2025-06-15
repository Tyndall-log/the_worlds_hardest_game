import sys
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from stage_map.env import Environment, SingleEnvironment
from model.model6 import CNNPolicy
from model.model7 import ResNetPolicy

Policy = ResNetPolicy

class EnvironmentVisualizer(QMainWindow):
	def __init__(self, checkpoint_path, fps=60):
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
		self.view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.setCentralWidget(self.view)

		# FPS 설정
		self.fps = fps
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_frame)
		self.timer.start(1000 // self.fps)

		# 사용자 액션 초기화
		self.user_action = 0
		self.move_up = False
		self.move_down = False
		self.move_left = False
		self.move_right = False
		s = 1
		self.movement_scale = s
		self.movement_to_action = {
			(0, 0): 0,
			(0, -s): 1,
			(s, -s): 2,
			(s, 0): 3,
			(s, s): 4,
			(0, s): 5,
			(-s, s): 6,
			(-s, 0): 7,
			(-s, -s): 8,
		}

		# 강화학습 환경 및 모델 초기화
		self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
		self.player_num = 1
		# self.env = Environment(
		# 	name="test",
		# 	map_path=Path(__file__).parent / "stage_map/stage/stage0/map1.json",
		# 	player_num=self.player_num,
		# 	fps=self.fps
		# )
		self.env = SingleEnvironment(
			name="model_play",
			map_path=Path(__file__).parent / "stage_map/stage/stage0/map1.json",
			fps=self.fps
		)
		self.env.render_mode = "rgb_array"
		self.env = DummyVecEnv([lambda: self.env])  # 벡터화된 환경으로 래핑
		self.n_stack = 3
		self.env = VecFrameStack(self.env, n_stack=self.n_stack, channels_order="first")  # 프레임 스택
		self.obs = self.env.reset()
		self.obs = torch.tensor(self.obs, dtype=torch.float32).to(self.device)
		self.model = Policy(4 * self.n_stack, 9).to(self.device)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		self.model.eval()
		self.cumulative_reward = 0

	def update_frame(self):
		# # 사용자 액션 설정
		# actions = [self.user_action] + [0] * (self.player_num - 1)

		# 모델 추론
		x = self.obs
		x /= 255.0  # Normalize the input
		logits, value = self.model(x)
		m = Categorical(logits=logits)
		actions = m.sample().cpu().numpy()

		# 환경 업데이트
		# obs, rewards, terminated, truncated, infos = self.env.step(actions)
		# done = terminated[0] or truncated[0]
		obs, rewards, done, infos = self.env.step(actions)
		self.obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
		self.cumulative_reward += rewards[0]

		if done:
			print(f"self.cumulative_reward: {self.cumulative_reward}")
			self.cumulative_reward = 0
			# self.obs = self.env.reset_player(0)[0]
			self.obs = self.env.reset()
			self.obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

		# 이미지 업데이트
		self.image_data = self.env.render()

		# NumPy 배열을 QImage로 변환
		height, width, channels = self.image_data.shape
		bytes_per_line = channels * width
		q_image = QImage(self.image_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

		# QPixmap으로 변환 후 QGraphicsPixmapItem으로 렌더링
		pixmap = QPixmap.fromImage(q_image)
		pixmap.setDevicePixelRatio(self.devicePixelRatioF())
		self.scene.clear()
		self.scene.addPixmap(pixmap)

	def keyPressEvent(self, event):
		if event.key() == Qt.Key.Key_Escape:
			self.close()
		if not event.isAutoRepeat():
			if event.key() in (Qt.Key.Key_W, Qt.Key.Key_Up):
				self.move_up = True
			elif event.key() in (Qt.Key.Key_S, Qt.Key.Key_Down):
				self.move_down = True
			elif event.key() in (Qt.Key.Key_A, Qt.Key.Key_Left):
				self.move_left = True
			elif event.key() in (Qt.Key.Key_D, Qt.Key.Key_Right):
				self.move_right = True
			self.update_user_action()

	def keyReleaseEvent(self, event):
		if not event.isAutoRepeat():
			if event.key() in (Qt.Key.Key_W, Qt.Key.Key_Up):
				self.move_up = False
			elif event.key() in (Qt.Key.Key_S, Qt.Key.Key_Down):
				self.move_down = False
			elif event.key() in (Qt.Key.Key_A, Qt.Key.Key_Left):
				self.move_left = False
			elif event.key() in (Qt.Key.Key_D, Qt.Key.Key_Right):
				self.move_right = False
			self.update_user_action()

	def update_user_action(self):
		user_x = self.move_right - self.move_left
		user_y = self.move_down - self.move_up
		self.user_action = self.movement_to_action.get((user_x, user_y), 0)

	def closeEvent(self, event):
		self.timer.stop()
		event.accept()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	visualizer = EnvironmentVisualizer(
		# checkpoint_path=Path(__file__).parent / "checkpoints/20250609_175535/model_step_465123.pt",
		checkpoint_path=Path(__file__).parent / "model_step_40.pt",
		fps=30,
	)
	visualizer.show()
	sys.exit(app.exec())