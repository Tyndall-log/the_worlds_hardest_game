import sys
import math
import json
import time
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.spaces import Box
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.a2c import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from stage_map.env import Environment, SingleEnvironment

root_path = Path(__file__).parent


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
		self.model_path = root_path / "model.pth"
		if self.model_path.exists():
			observation_space = gym.spaces.Box(low=0, high=255, shape=(4*3, 160, 256), dtype=np.uint8)
			action_space = gym.spaces.Discrete(9)
			self.policy = CnnPolicy(
				observation_space=observation_space,
				action_space=action_space,
				lr_schedule=lambda _: 1e-3,
			)
			self.policy.load_state_dict(torch.load(self.model_path, map_location=self.device))
		else:
			s = input("모델을 찾을 수 없습니다. 모델 없이 실행하시겠습니까? (y/n): ")
			if s.lower() != 'y':
				sys.exit(0)
		self.player_num = 1
		self.auto_play = False
		self.map_path = root_path / "export_train/base_map.json"
		self.save_path = root_path / f"export_train/raw_data/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
		with open(self.map_path.as_posix(), 'r') as f:
			self.map_data_dict = json.load(f)
			self.map_data_dict_temp = deepcopy(self.map_data_dict)
			self.randomize_map()
		self.env = SingleEnvironment(
			name="make_raw_data",
			# map_path=self.map_path,
			map_data_dict=self.map_data_dict_temp,
			fps=self.fps,
		)
		self.env = DummyVecEnv([lambda: self.env])  # 벡터화된 환경으로 래핑
		self.n_stack = 3
		self.env = VecFrameStack(self.env, n_stack=self.n_stack, channels_order="first")  # 프레임 스택
		obs = self.env.reset()
		self.map_data_dict_temp['player_info'] = {
			"init_pos": [
				int(self.env.envs[0].player_object_list[0].pos.x),
				int(self.env.envs[0].player_object_list[0].pos.y),
			],
		}
		self.obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
		self.image_data = self.env.envs[0].render()
		self.draw_image(self.image_data)
		self.cumulative_reward = 0
		self.reward_list = []
		self.episode_actions = []
		self.episode_trajectories = []
		self.episode_start_flag = False
		self.skip_flag = False
		self.auto_one_flag = False

		# === 단축키 안내 ===
		print("단축키 안내:")
		print("W, ↑: 위로 이동")
		print("S, ↓: 아래로 이동")
		print("A, ←: 왼쪽으로 이동")
		print("D, →: 오른쪽으로 이동")
		print("Space: 에피소드 시작/건너뛰기(움직이려면 반드시 Space를 눌러야 함)")
		print("R: 에피소드 리셋")
		print("T: 모델 자동 플레이 토글")
		print("Q: 모델 자동 플레이 한 번 실행")

	def reset_episode(self):
		self.randomize_map()
		# self.env = Environment(
		# 	name="make_raw_data",
		# 	# map_path=self.map_path,
		# 	map_data_dict=self.map_data_dict_temp,
		# 	player_num=self.player_num,
		# 	fps=self.fps,
		# )
		self.env.envs[0].reset_map(self.map_data_dict_temp)
		obs = self.env.reset()
		self.map_data_dict_temp['player_info'] = {
			"init_pos": [
				int(self.env.envs[0].player_object_list[0].pos.x),
				int(self.env.envs[0].player_object_list[0].pos.y),
			],
		}
		self.obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
		self.image_data = self.env.envs[0].render()
		self.draw_image(self.image_data)
		self.cumulative_reward = 0
		self.reward_list = []
		self.episode_actions = []
		self.episode_start_flag = False
		self.skip_flag = False
		self.auto_one_flag = False

	def randomize_map(self):
		"""
		맵 데이터를 무작위로 변경합니다.
		"""
		temp = deepcopy(self.map_data_dict)
		# matrix = np.array(temp["matrix"])
		# ground_mask = matrix == 1

		# 설정
		coin_count = int(np.clip(np.random.normal(loc=5, scale=1), 1, 20))  # 최소 1, 최대 20개
		coin_size = 50
		coin_radius = coin_size / 2

		coin_x1, coin_y1, coin_x2, coin_y2 = 300, 300, 800, 500

		# 배치 가능한 범위 (반지름 고려)
		xmin = coin_x1 + coin_radius
		xmax = coin_x2 - coin_radius
		ymin = coin_y1 + coin_radius
		ymax = coin_y2 - coin_radius

		positions = []

		max_attempts = 1000
		attempt = 0

		while len(positions) < coin_count and attempt < max_attempts:
			attempt += 1
			x = round(np.random.uniform(xmin, xmax) / 5) * 5  # 5의 배수로 반올림
			y = round(np.random.uniform(ymin, ymax) / 5) * 5  # 5의 배수로 반올림

			# 겹치는지 검사
			overlap = False
			for (px, py) in positions:
				dist = np.hypot(px - x, py - y)
				if dist < coin_size:  # 반지름 * 2
					overlap = True
					break

			if not overlap:
				positions.append((round(x), round(y)))

		if len(positions) < coin_count:
			print(f"⚠️ {coin_count}개 중 {len(positions)}개만 배치됨 (공간 부족)")

		temp["coin_list"] = [
			{
				"pos": [pos[0], pos[1]],
			}
			for pos in positions
		]

		self.map_data_dict_temp = temp

	def update_frame(self):
		if self.auto_play or self.auto_one_flag:
			self.auto_one_flag = False
			# 자동 플레이: policy가 행동 예측
			# obs_for_policy = self.obs.cpu().numpy()  # policy는 numpy 입력을 원함
			# action, _ = self.policy.predict(obs_for_policy, deterministic=True)
			distribution = self.policy.get_distribution(self.obs.cpu())
			probs = distribution.distribution.probs  # 행동별 확률
			log_probs = distribution.distribution.logits
			print(f"Action probabilities: {probs[0]}")
			# action = distribution.get_actions(deterministic=True)
			action = distribution.get_actions()
			action = action[0].item()  # 벡터화된 환경에서 첫 번째 에이전트의 행동을 가져옴
		else:
			action = self.user_action
			if not self.episode_start_flag:
				return
			if action == 0 and not self.skip_flag:
				return

		# 환경 업데이트
		self.episode_actions.append(action)
		# obs, rewards, terminated, truncated, infos = self.env.step(actions)
		# done = terminated[0] or truncated[0]
		obs, rewards, done, infos = self.env.step([action])
		self.obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
		self.cumulative_reward += rewards[0]
		self.reward_list.append(rewards[0])
		print(rewards[0])

		if done[0]:
			self.episode_start_flag = False
			self.episode_trajectories.append({
				"map_data": self.map_data_dict_temp,
				"actions": self.episode_actions.copy(),
			})
			self.episode_actions.clear()
			# with open(self.save_path, 'w') as f:
			# 	json.dump(self.episode_trajectories, f, indent=4)
			print(f"self.cumulative_reward: {self.cumulative_reward}")
			self.cumulative_reward = 0
			# self.obs = self.env.envs[0].reset_player(0)
			self.obs = self.env.reset()
			print(f"Resetting environment, cumulative reward: {self.cumulative_reward}")
			print(f"Rewards in this episode: {self.reward_list}")
			self.reward_list.clear()
			self.reset_episode()


		# 이미지 업데이트
		self.image_data = self.env.envs[0].render()
		if self.image_data is not None:
			self.draw_image(self.image_data)

	def draw_image(self, image):
		"""
		이미지를 QGraphicsScene에 그립니다.
		:param image: NumPy 배열 형태의 이미지
		"""
		height, width, channels = image.shape
		bytes_per_line = channels * width
		q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
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
			elif event.key() == Qt.Key.Key_Space:
				self.episode_start_flag = True
				self.skip_flag = True
			elif event.key() == Qt.Key.Key_R:
				self.reset_episode()
			elif event.key() == Qt.Key.Key_T:
				self.auto_play = not self.auto_play
			elif event.key() == Qt.Key.Key_Q:
				self.auto_one_flag = True
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
			elif event.key() == Qt.Key.Key_Space:
				self.skip_flag = False
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
	visualizer = EnvironmentVisualizer(fps=30)
	visualizer.show()
	sys.exit(app.exec())