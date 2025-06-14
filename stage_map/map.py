from pathlib import Path
import json
from collections import deque

import cv2
import numpy as np

from stage_map.mask import MaskInfo
from stage_map.object.utils import Point, ObjectManager
from stage_map.object.player import Player
from stage_map.object.ball import Ball
from stage_map.object.coin import Coin


class MapData:
	class ThemeInfo:
		def __init__(self, theme_name: str):
			if theme_name == "default" or theme_name == "blue":
				self.letterbox_color = (0, 0, 0)
				self.background_color = (180, 181, 254)
				self.floor_tile_color_list = [(247, 247, 255), (230, 230, 255)]
				self.ball_tile_color = (0, 0, 255)
				self.checkpoint_zone_color = (182, 254, 180)
			else:
				raise ValueError(f"지원하지 않는 테마: {theme_name}")
			self.theme_name = theme_name

		def __str__(self):
			return f"ThemeInfo(\"{self.theme_name}\")"

	map_ready_flag: bool
	map_read_only_flag: bool = True
	map_name: str = "아직 불러오지 않음"
	map_theme: ThemeInfo = ThemeInfo("default")
	map_path: Path
	map_width: int = 20
	map_height: int = 12
	map_file_name: str
	image_width: int = 1100
	image_height: int = 800
	player_info: dict
	ball_data_list: list[dict]
	# _checkpoint_grid: list[list[int]] = [[0] * (map_width + 2) for _ in range(map_width + 2)]  # padding
	# _checkpoint_grid: np.array = np.zeros((map_height + 2, map_width + 2), dtype=np.uint8)
	# _checkpoint_count: int = 0
	# _checkpoint_respon_pos: list[Point] = []
	# _goal_num: int = 0

	def __init__(self, map_file_path: Path | None = None, map_data_dict: dict | None = None, map_read_only_flag: bool = True):
		self.map_file_path = map_file_path
		self.map_data_dict = map_data_dict
		self.map_file_name = map_file_path.name if map_file_path is not None else "아직 불러오지 않음"
		self.object_manager: ObjectManager = ObjectManager()
		self.background_image: np.ndarray = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
		self.mask_info: MaskInfo = MaskInfo(np.zeros((self.image_height, self.image_width), dtype=np.uint8))
		self.ball_data_list = []
		self.coin_data_list = []
		self._checkpoint_count = 0
		self._checkpoint_id_grid = np.zeros((self.map_height + 2, self.map_width + 2), dtype=np.uint8)
		self._checkpoint_id_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
		self._coin_count = 0
		self.map_ready_flag = False
		if map_file_path is not None:
			with open(self.map_file_path, 'r') as file:
				data = json.load(file)
				self.load_map(data)
		elif map_data_dict is not None:
			self.map_data_dict = map_data_dict
			self.load_map(self.map_data_dict)
		else:
			raise ValueError("맵 파일 경로 또는 맵 데이터 딕셔너리를 제공해야 합니다.")

	def reset(self, map_file_path: Path | None = None, map_data_dict: dict | None = None):
		"""맵 데이터를 초기화합니다."""
		self.object_manager = ObjectManager()
		self.ball_data_list.clear()
		self.coin_data_list.clear()
		self.map_ready_flag = False
		if map_file_path is not None:
			self.map_file_path = map_file_path
			self.map_file_name = map_file_path.name
			with open(self.map_file_path, 'r') as file:
				data = json.load(file)
				self.load_map(data)
		elif map_data_dict is not None:
			self.map_data_dict = map_data_dict
			self.load_map(self.map_data_dict)
		else:
			raise ValueError("맵 파일 경로 또는 맵 데이터 딕셔너리를 제공해야 합니다.")

	def draw_background(self, data: dict):
		map_matrix = np.array(data["matrix"])
		if map_matrix.shape != (self.map_height, self.map_width):
			raise ValueError("맵 크기가 12x20이 아닙니다.")
		padded_matrix = np.pad(map_matrix, ((1, 1), (1, 1)), mode='constant', constant_values=0)
		map_height = self.map_height
		map_width = self.map_width

		b_img = self.background_image
		m_img = self.mask_info.mask_image
		b_img[:53, :] = self.map_theme.letterbox_color
		m_img[:53, :] = MaskInfo.MaskLayer.LETTERBOX
		b_img[53:-53, :] = self.map_theme.background_color
		m_img[53:-53, :] = MaskInfo.MaskLayer.WALL
		b_img[-53:, :] = self.map_theme.letterbox_color
		m_img[-53:, :] = MaskInfo.MaskLayer.LETTERBOX
		grid_size = 50
		# shift = 10

		# 체크포인트 지역 분할
		checkpoint_id_grid = self._checkpoint_id_grid
		next_checkpoint_id = 0

		def checkpoint_bfs(i, j):
			# if self._checkpoint_grid[i, j] == 0:
			if padded_matrix[i, j] != 2:
				return
			queue = deque([(i, j)])
			while queue:
				x, y = queue.popleft()
				if checkpoint_id_grid[x, y] != 0:
					continue
				checkpoint_id_grid[x, y] = next_checkpoint_id
				for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
					nx, ny = x + dx, y + dy
					if padded_matrix[nx][ny] == 2:
						queue.append((nx, ny))

		for y in range(1, map_height + 1):
			for x in range(1, map_width + 1):
				if padded_matrix[y][x] == 2 and checkpoint_id_grid[y][x] == 0:
					next_checkpoint_id += 1
					checkpoint_bfs(y, x)
		self._checkpoint_count = next_checkpoint_id

		# 바닥 타일 배치 & 체크포인트 지역 타일 배치
		checkpoint_id_mask = self._checkpoint_id_mask
		for y in range(1, map_height + 1):
			for x in range(1, map_width + 1):
				if padded_matrix[y][x] == 0:
					continue
				y1 = 50 + y * grid_size
				x1 = x * grid_size
				if padded_matrix[y][x] == 1:
					tile_color = self.map_theme.floor_tile_color_list[(x + y) % 2]
					b_img[y1:y1 + grid_size, x1:x1 + grid_size] = tile_color
					m_img[y1:y1 + grid_size, x1:x1 + grid_size] = 0
				elif padded_matrix[y][x] == 2:
					tile_color = self.map_theme.checkpoint_zone_color
					b_img[y1:y1 + grid_size, x1:x1 + grid_size] = tile_color
					m_img[y1:y1 + grid_size, x1:x1 + grid_size] = MaskInfo.MaskLayer.CHECKPOINT_ZONE
					checkpoint_id_mask[y1:y1 + grid_size, x1:x1 + grid_size] = checkpoint_id_grid[y][x]
				else:
					raise ValueError(f"알 수 없는 타일 값: {padded_matrix[y][x]}")

		# 벽(경계) 추가
		for y in range(0, map_height + 1):
			for x in range(0, map_width + 1):
				g1 = padded_matrix[y][x]
				g2 = padded_matrix[y][x + 1]
				g3 = padded_matrix[y + 1][x]
				g4 = padded_matrix[y + 1][x + 1]
				if (0 < g1) ^ (0 < g2):
					y1 = 50 + y * grid_size - 3
					y2 = 50 + (y + 1) * grid_size + 3
					x1 = (x + 1) * grid_size - 3
					x2 = (x + 1) * grid_size + 3
					b_img[y1:y2, x1:x2] = (0, 0, 0)
					m_img[y1:y2, x1:x2] = 0
				if (0 < g1) ^ (0 < g3):
					y1 = 50 + (y + 1) * grid_size - 3
					y2 = 50 + (y + 1) * grid_size + 3
					x1 = x * grid_size - 3
					x2 = (x + 1) * grid_size + 3
					b_img[y1:y2, x1:x2] = (0, 0, 0)
					m_img[y1:y2, x1:x2] = 0

		# if g1 == 0 and g2 == 0 and g3 == 0 and 0 < g4:
		# 	y1 = 53 + (y + 1) * grid_size - 3
		# 	y2 = 53 + (y + 1) * grid_size + 3
		# 	x1 = (x + 1) * grid_size - 3
		# 	x2 = (x + 1) * grid_size + 3
		# 	b_img[y1:y2, x1:x2] = (0, 0, 0)
		# 	m_img[y1:y2, x1:x2] = MaskInfo.MaskLayer.WALL
		return

	def load_map(self, data: dict):
		if "map_name" not in data:
			raise KeyError(f"{Path}에 필수 키 \"map_name\"이(가) 존재하지 않습니다.")
		self.map_name = data["map_name"]
		if "map_theme" in data:
			self.map_theme = self.ThemeInfo(data["map_theme"])
		if "matrix" not in data:
			raise KeyError(f"{Path}에 필수 키 \"matrix\"이(가) 존재하지 않습니다.")
		self.draw_background(data)

		if "player_info" in data:
			self.player_info = data["player_info"]
		if "ball_list" in data:
			self.ball_data_list = data["ball_list"]
		if "coin_list" in data:
			self.coin_data_list = data["coin_list"]
			self._coin_count = len(self.coin_data_list)
		self.map_ready_flag = True

	def get_player(self, player_id: int, random_init_pos: bool = True):
		return Player(
			object_manager=self.object_manager,
			player_id=player_id,
			data=self.player_info,
			random_init_pos=random_init_pos,
		)

	def get_ball_data(self):
		ball_object_list = []
		for data in self.ball_data_list:
			ball_object_list.append(Ball(
				object_manager=self.object_manager,
				data=data,
			))
		return ball_object_list

	def get_coin_data(self):
		coin_object_list = []
		for data in self.coin_data_list:
			coin_object_list.append(Coin(
				object_manager=self.object_manager,
				data=data,
			))
		return coin_object_list

	@property
	def checkpoint_map_grid(self):
		return self._checkpoint_id_grid[1:-1, 1:-1]  # padding 제거

	@property
	def checkpoint_id_mask(self):
		return self._checkpoint_id_mask

	@property
	def checkpoint_count(self):
		return self._checkpoint_count

	@property
	def coin_count(self):
		return self._coin_count

# class MapEnv:
# 	def __init__(self, player_count: int):


if __name__ == "__main__":
	map_data = MapData(Path(__file__).parent / "map_file" / "map1.json")
	print(map_data.map_name)
	print(map_data.map_theme)
	# RGA to BGR
	background_image = cv2.cvtColor(map_data.background_image, cv2.COLOR_RGB2BGR)
	cv2.imshow("Map", background_image)
	# mask_image = cv2.cvtColor(map_data.mask_info.mask_image, cv2.COLOR_)
	mask_image = map_data.mask_info.mask_image.squeeze(2)
	mask_image_3 = np.zeros((800, 1100, 3), dtype=np.uint8)
	mask_image_3[mask_image == 0, :] = (200, 200, 200)
	mask_image_3[mask_image == MaskInfo.MaskLayer.WALL, :] = (0, 0, 0)
	mask_image_3[mask_image == MaskInfo.MaskLayer.CHECKPOINT_ZONE, :] = (0, 255, 0)

	cv2.imshow("Mask", mask_image_3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
