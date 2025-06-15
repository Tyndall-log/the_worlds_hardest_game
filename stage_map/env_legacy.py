from pathlib import Path
import time

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from matplotlib.pyplot import imshow

from stage_map.object.utils import EnvData, ObjectManager
from stage_map.object.player import Player
from stage_map.object.ball import Ball
from stage_map.object.coin import Coin
from stage_map.map import MapData

class Environment(gym.Env):
	_map_data: MapData
	_object_manager: ObjectManager
	_env_data: EnvData
	# _env_data_list: list[EnvData]
	# _frame_per_second: int = 30
	# _current_frame: int = 0
	observation: np.ndarray
	orign_observation: np.ndarray

	render_width = 1100
	render_height = 800
	crop_width = 1024
	crop_height = 640
	letterbox_color = (0, 0, 0)  # 배경색
	letterbox = 53  # 상하단 여백
	crop_offset_x = (render_width - crop_width) // 2
	crop_offset_y = (render_height - crop_height) // 2
	input_width = crop_width // 4
	input_height = crop_height // 4

	def __init__(
		self,
		name: str,
		map_path: Path = Path(__file__).parent / "map_file" / "map1.json",
		player_num: int = 16,
		fps: int = 30,
		max_time_seconds: int = 60,
		train_mode: bool = True,
	):
		super(Environment, self).__init__()
		self.name = name  # 환경 이름

		self.player_num = player_num  # 플레이어 수 (배치 크기와 동일)
		self.map_path = map_path  # 맵 파일 경로

		# 행동 공간: 정지, 상, 우상, 우, 우하, 하, 좌하, 좌, 좌상
		self.action_space = spaces.Discrete(9)

		# 관찰 공간: 4채널 이미지 (4배 해상도 감소)
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(player_num, 4, self.input_height, self.input_width),
			dtype=np.uint8
		)
		self.observation = np.zeros((player_num, self.input_height, self.input_width, 4), dtype=np.uint8)

		# 맵 데이터 로드
		self._map_data = MapData(self.map_path)

		# 객체 관리자 초기화
		self._object_manager = self._map_data.object_manager

		# 최대 스텝 수
		self.max_step = int(max_time_seconds * fps)

		# 환경 정보
		# self.env_data = EnvData(self._map_data.background_image, self._map_data.mask_info.mask_image)

		self.train_mode = train_mode  # 학습 모드 여부

		# 플레이어 오브젝트 준비
		self.player_object_list = [
			self._map_data.get_player(i, random_init_pos=True) for i in range(self.player_num)
		]

		# 공 오브젝트 준비
		self.ball_object_list = self._map_data.get_ball_data()

		# 코인 오브젝트 준비
		self.coin_object_list = self._map_data.get_coin_data()

		# 환경 정보
		self._env_data = EnvData(
			name=self.name,
			object_manager=self._object_manager,
			player_object_list=self.player_object_list,
			coin_object_list=self.coin_object_list,
			map_data=self._map_data,
		)


	def reset(self, seed=None, options=None):
		"""
		환경을 초기화하고 초기 상태를 반환.
		"""
		super().reset(seed=seed)

		for p in self.player_object_list:
			p.reset(env_data=self._env_data, seed=seed)
		self._objects_draw()

		self.orign_observation = np.concatenate([
			self._env_data.player_draw_canvas,
			self._env_data.player_trail_canvas[..., np.newaxis]
		], axis=-1)
		self.origin_image_random_crop_resize(
			self.orign_observation,
			self.observation,
			crop_offset=(self.crop_offset_x, self.crop_offset_y),
		)

		return self.observation, {}

	def reset_player(self, player_id: int):
		"""
		특정 플레이어를 초기화하고 초기 상태를 반환.
		Args:
			player_id (int): 초기화할 플레이어의 ID.
		"""
		if 0 <= player_id < self.player_num:
			self.player_object_list[player_id].reset(env_data=self._env_data)
			self._objects_draw()
			temp = np.zeros((1, self.input_height, self.input_width, 3), dtype=np.uint8)
			self.origin_image_random_crop_resize(
				self._env_data.player_draw_canvas[player_id:player_id + 1],
				temp,
				crop_offset=(self.crop_offset_x, self.crop_offset_y),
			)
			temp2 = np.zeros((1, self.input_height, self.input_width, 1), dtype=np.uint8)
			self.origin_image_random_crop_resize(
				self._env_data.player_trail_canvas[player_id:player_id + 1, ..., np.newaxis],
				temp2,
				crop_offset=(self.crop_offset_x, self.crop_offset_y),
			)
			obs = np.concatenate([
				temp,
				temp2,
			], axis=-1)
			return obs[0], {}
		else:
			raise ValueError(f"Invalid player_id: {player_id}. Must be between 0 and {self.player_num - 1}.")

	def _objects_step(self):
		# for ball in self.ball_object_list:
		# 	ball.step(self._env_data)
		# for player in self.player_object_list:
		# 	env_data = self._env_data_list[i]
		# 	env_data.collision_mask[...] = self._env_data.collision_mask
		# 	player.step(self._env_data)
		self._object_manager.step(self._env_data)

	def _objects_draw(self):
		# for ball in self.ball_object_list:
		# 	ball.draw(self._env_data)
		# for i, player in enumerate(self.player_object_list):
		# 	env_data = self._env_data_list[i]
		# 	env_data.canvas[...] = self._env_data.canvas
		# 	player.draw(env_data)
		self._object_manager.draw(self._env_data)

	def step(self, actions):
		"""
		환경을 한 스텝 진행하고 다음 상태, 보상, 종료 여부, 추가 정보를 반환.

		Args:
			actions (np.array): 배치 크기만큼의 행동 배열.

		Returns:
			observation (np.array): 다음 상태.
			rewards (np.array): 보상 배열.
			dones (np.array): 종료 여부 배열.
			infos (dict): 추가 정보.
		"""
		# 플레이어 위치 업데이트
		for i in range(self.player_num):
			action = actions[i]
			player = self.player_object_list[i]
			player.action = action

		# 다음 상태 관찰값 생성 (예: 이미지 데이터)
		self._env_data.reset()
		self._env_data.current_frame += 1
		self._objects_step()
		self._objects_draw()
		temp = np.zeros((self.player_num, self.input_height, self.input_width, 3), dtype=np.uint8)
		self.origin_image_random_crop_resize(
			self._env_data.player_draw_canvas,
			temp,
			crop_offset=(self.crop_offset_x, self.crop_offset_y),
		)
		temp2 = np.zeros((self.player_num, self.input_height, self.input_width, 1), dtype=np.uint8)
		self.origin_image_random_crop_resize(
			self._env_data.player_trail_canvas[..., np.newaxis],
			temp2,
			crop_offset=(self.crop_offset_x, self.crop_offset_y),
		)
		self.observation = np.concatenate([
			temp,
			temp2,
		], axis=-1)

		# 보상 계산
		rewards = []
		for i in range(self.player_num):
			rewards.append(self.player_object_list[i].reward)

		# 종료 여부
		terminated = np.zeros(self.player_num, dtype=bool)
		for i in range(self.player_num):
			player = self.player_object_list[i]
			terminated[i] = player.is_dead or player.is_goal
			# if terminated[i]:
			# 	player.reset(env_data=self._env_data)

		# 중단 여부
		truncated = np.zeros(self.player_num, dtype=bool)
		for i in range(self.player_num):
			player = self.player_object_list[i]
			if self.max_step <= player.step_count:
				truncated[i] = True

				# # 최종 보상 계산
				# coin_count = self._env_data.coin_count
				# coin_collected = len(player.coin_id_set)
				# sum_count = coin_count + 2  # 코인 개수 + 2 (최대 보상 계산을 위한 상수)
				# success_ratio = coin_collected / sum_count
				#
				# # 실패에 대한 보상 (예: -goal_reward ~ +goal_reward)
				# final_reward = -player.goal_reward + 2 * player.goal_reward * success_ratio
				# final_reward += player.step_penalty * player.step_count
				# rewards[i] += final_reward

				# player.reset(env_data=self._env_data)

		# 추가 정보
		infos = {}

		return self.observation.copy(), rewards, terminated, truncated, infos

	def render(self, mode="high_rgb_array"):
		"""
		환경 시각화를 렌더링
		"""
		if mode == "human":
			cv2.imshow("Map", self._env_data.init_canvas)
			cv2.waitKey(1)
		elif mode == "high_rgb_array":
			# for ball in self.ball_object_list:
			# 	ball.draw(self._env_data)
			canvas = self._env_data.dynamic_canvas.copy()
			for player in self.player_object_list:
				player.draw(self._env_data, view_canvas=canvas)
			for coin in self.coin_object_list:
				coin.draw_rendering(self._env_data, view_canvas=canvas)
			return canvas
			# return self._env_data_list[0].canvas
		elif mode == "rgb_array":
			return self.observation

	def close(self):
		"""
		환경 종료 시 호출.
		"""
		pass


	@staticmethod
	def origin_image_random_crop_resize(
		image: np.ndarray,
		dst_batch_image: np.ndarray,
		crop_offset: tuple[int, int] = (-1, -1),
		crop_size: tuple[int, int] = (crop_width, crop_height),
		resize_size: tuple[int, int] = (input_width, input_height),
	):
		letter_box = Environment.letterbox
		batch_size = dst_batch_image.shape[0]

		image_width, image_height = image.shape[2], image.shape[1]
		crop_offsets_x = np.random.randint(0, image_width - crop_size[0], size=batch_size)\
			if crop_offset[0] == -1 else np.full(batch_size, crop_offset[0])
		crop_offsets_y = np.random.randint(letter_box, image_height - crop_size[1] - letter_box, size=batch_size)\
			if crop_offset[1] == -1 else np.full(batch_size, crop_offset[1])

		for i in range(batch_size):
			cropped_image = image[
				i,
				crop_offsets_y[i]:crop_offsets_y[i] + crop_size[1],
				crop_offsets_x[i]:crop_offsets_x[i] + crop_size[0]
			]
			cv2.resize(cropped_image, resize_size, dst=dst_batch_image[i], interpolation=cv2.INTER_AREA)


# 테스트 코드
if __name__ == "__main__":
	import time
	player_num = 16
	fps: int = 30
	env = Environment(
		map_path=Path(__file__).parent / "map_file" / "map2.json",
		player_num=player_num,
		fps=fps,
	)
	cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	user_action = 0
	s = 1
	movement_to_action = {
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

	# 초기 시간 기록
	start_time = time.time()
	for i in range(int(1e10)):

		if i == 0:
			obs, _ = env.reset()
			# print(f"Initial Observation Shape: {obs.shape}")
		else:
			t1 = time.time()
			actions = np.random.randint(0, 9, size=player_num)  # 랜덤 행동
			actions[0] = user_action  # 첫 번째 플레이어는 사용자 입력 행동
			obs, rewards, terminated, truncated, infos = env.step(actions)
			t2 = time.time()
			print(f"Step Time(i: {i}): {(t2 - t1) * 1000:.3f}ms")
			# print(f"Next Observation Shape: {obs.shape}")
			# print(f"Rewards: {rewards}")
			# print(f"Dones: {dones}")

		# 렌더링 처리
		image = env.render()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		h, w = image.shape[:2]
		cv2.imshow("image", image)

		# 대기 시간 계산
		stop = False
		first_flag = True
		while True:  # 남은 시간이 있으면 대기
			key = cv2.waitKey(1)
			# time.sleep(1000000)
			if key == 27:  # ESC 키로 종료
				stop = True
				break
			elif key == ord("a"):
				start_time += 1 / fps
			elif key == ord("d"):
				start_time -= 1 / fps
			elif key == ord("s"):
				time.sleep(0.5)
			# 현재 프레임 처리 시간 계산
			elapsed_time = time.time() - start_time - i / fps
			delay_time = 1 / fps - elapsed_time
			if delay_time <= 0:
				if first_flag:
					print(f"프레임 처리 시간이 목표 FPS를 초과했습니다: {elapsed_time*1000:.3f}ms")
				break
			else:
				first_flag = False

		if stop:
			break

	cv2.destroyAllWindows()
