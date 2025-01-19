import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import threading
from typing import ClassVar

from stage_map.object.utils import Point
from stage_map.map import MapData


class Environment(gym.Env):
	_id: int
	_map_data: MapData

	def __init__(self):
		super().__init__()

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# return observation, info

	def step(self, action):
		pass
		# return observation, reward, done, truncated, info

	def render(self):
		if self.render_mode == "human":
			pass


class CustomEnv(gym.Env):
	metadata = {"render_modes": ["human"], "render_fps": 30}

	def __init__(self, render_mode=None, grid_size=5):
		super().__init__()

		# 환경 속성
		self.grid_size = grid_size  # 격자 크기
		self.agent_pos = np.array([0, 0], dtype=np.int32)
		self.goal_pos = np.array([grid_size - 1, grid_size - 1])  # 목표 위치
		self.steps = 0  # 현재 스텝 수
		self.max_steps = 50  # 최대 스텝 수

		# 행동 공간: 상, 하, 좌, 우
		self.action_space = spaces.Discrete(4)

		# 관측 공간: 에이전트와 목표의 위치 (x, y)
		self.observation_space = spaces.Box(
			low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
		)

		# 렌더링 모드 설정
		self.render_mode = render_mode
		self.window = None

	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# 초기화
		self.agent_pos = np.array([0, 0])
		self.steps = 0
		observation = self._get_obs()
		info = {"goal_pos": self.goal_pos}

		return observation, info

	def step(self, action):
		# 행동 수행 (상, 하, 좌, 우)
		if action == 0:  # 상
			self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
		elif action == 1:  # 하
			self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
		elif action == 2:  # 좌
			self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
		elif action == 3:  # 우
			self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

		# 보상 계산
		reward = 1 if np.array_equal(self.agent_pos, self.goal_pos) else -0.1

		# 종료 조건
		self.steps += 1
		done = np.array_equal(self.agent_pos, self.goal_pos)  # 목표에 도달하면 종료
		truncated = self.steps >= self.max_steps  # 최대 스텝 초과

		# 관측값과 추가 정보 반환
		observation = self._get_obs()
		info = {}

		return observation, reward, done, truncated, info

	def render(self):
		if self.render_mode == "human":
			if self.window is None:
				self._init_render()

			grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
			grid[:] = "."
			grid[self.goal_pos[1], self.goal_pos[0]] = "G"  # 목표
			grid[self.agent_pos[1], self.agent_pos[0]] = "A"  # 에이전트
			print("\n".join(" ".join(row) for row in grid))
			print("\n")

	def _get_obs(self):
		return self.agent_pos.copy()

	def _init_render(self):
		pass  # 필요시 OpenCV, Pygame 등으로 시각화 초기화 가능