from pathlib import Path

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from stage_map.object.utils import EnvData
from stage_map.object.player import Player
from stage_map.object.ball import Ball
from stage_map.map import MapData


class Environment(gym.Env):
	_map_data: MapData
	_env_data: EnvData
	_env_data_list: list[EnvData]
	# _frame_per_second: int = 30
	# _current_frame: int = 0
	observation: np.ndarray
	orign_observation: np.ndarray

	render_width = 1100
	render_height = 800
	crop_width = 1024
	crop_height = 640
	input_width = crop_width // 4
	input_height = crop_height // 4

	def __init__(
		self,
		map_path: Path = Path(__file__).parent / "map_file" / "map1.json",
		batch_size: int = 16,
		fps: int = 30,
	):
		super(Environment, self).__init__()

		self.player_object_list: list[Player] = []
		self.ball_object_list: list[Ball] = []

		self.batch_size = batch_size  # 플레이어 수 (배치 크기와 동일)
		self.map_path = map_path  # 맵 파일 경로

		# 행동 공간: 상, 우상, 우, 우하, 하, 좌하, 좌, 좌상, 정지
		self.action_space = spaces.Discrete(9)

		# 관찰 공간: 3채널 이미지 (640x1152) (4배 해상도 감소)
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(batch_size, 3, self.input_height, self.input_width),
			dtype=np.uint8
		)
		self.observation = np.zeros((batch_size, self.input_height, self.input_width, 3), dtype=np.uint8)

		# 맵 데이터 로드
		self._map_data = MapData(self.map_path)

		# 환경 정보
		# self.env_data = EnvData(self._map_data.background_image, self._map_data.mask_info.mask_image)

	def reset(self, seed=None, options=None):
		"""
		환경을 초기화하고 초기 상태를 반환.
		"""
		super().reset(seed=seed)

		# 맵 데이터 로드
		self._map_data = MapData(self.map_path)

		# 플레이어 오브젝트 준비
		self.player_object_list = [
			self._map_data.get_player() for _ in range(self.batch_size)
		]

		# 공 오브젝트 준비
		self.ball_object_list = self._map_data.get_ball_data()

		# 환경 정보
		self._env_data = EnvData(self._map_data.background_image.copy(), self._map_data.mask_info.mask_image.copy())
		self.orign_observation = np.repeat(self._map_data.background_image[np.newaxis, :, :, :], repeats=16, axis=0)
		self._env_data_list = [
			EnvData(self.orign_observation[i], self._env_data.collision_mask)
			for i in range(self.batch_size)
		]
		self._objects_draw()
		self.origin_image_random_crop_resize(self.orign_observation, self.observation)
		# cv2.imshow("Map", self._env_data.canvas)

		return self.observation, {}

	def _objects_step(self):
		for ball in self.ball_object_list:
			ball.step(self._env_data)
		for player in self.player_object_list:
			player.step(self._env_data)

	def _objects_draw(self):
		for ball in self.ball_object_list:
			ball.draw(self._env_data)
		for i, player in enumerate(self.player_object_list):
			env_data = self._env_data_list[i]
			env_data.canvas[:, :, :] = self._env_data.canvas
			# env_data.collision_mask[:, :] = self._env_data.collision_mask
			player.draw(env_data)

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
		for i in range(self.batch_size):
			action = actions[i]
			player = self.player_object_list[i]
			player.action = action

		# 다음 상태 관찰값 생성 (예: 이미지 데이터)
		self._env_data.set(self._map_data.background_image.copy(), self._map_data.mask_info.mask_image.copy())
		self._env_data.current_frame += 1
		self._objects_step()
		self._objects_draw()
		self.origin_image_random_crop_resize(self.orign_observation, self.observation)

		# 보상 계산
		for i in range(self.batch_size):
			rewards[i] = self.player_object_list[i].reward

		# 종료 여부 (임시로 False로 설정)
		terminated = np.zeros(self.batch_size, dtype=bool)
		# for i in range(self.batch_size):
		#     terminated[i] = ...  # 종료 조건 추가

		# 종료 여부 (임시로 False로 설정)
		truncated = np.zeros(self.batch_size, dtype=bool)
		# for i in range(self.batch_size):
		#     truncated[i] = ...  # 종료 조건 추가

		# 추가 정보
		infos = {}

		return self.observation, rewards, terminated, truncated, infos

	def render(self, mode="high_rgb_array"):
		"""
		환경 시각화를 렌더링
		"""
		if mode == "human":
			cv2.imshow("Map", self._env_data.canvas)
			cv2.waitKey(1)
		elif mode == "high_rgb_array":
			# for ball in self.ball_object_list:
			# 	ball.draw(self._env_data)
			for player in self.player_object_list:
				player.draw(self._env_data)
			return self._env_data.canvas
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
		letter_box = 53
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
	env = Environment(batch_size=player_num, fps=fps)
	cv2.namedWindow("image", cv2.WINDOW_NORMAL)

	# 초기 시간 기록
	start_time = time.time()
	for i in range(int(1e10)):

		if i == 0:
			obs, _ = env.reset()
			# print(f"Initial Observation Shape: {obs.shape}")
		else:
			t1 = time.time()
			actions = np.random.randint(0, 9, size=player_num)  # 랜덤 행동
			obs, rewards, dones, infos = env.step(actions)
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
