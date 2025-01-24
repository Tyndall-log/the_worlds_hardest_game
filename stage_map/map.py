from pathlib import Path
import numpy as np
import json
import cv2

from stage_map.mask import MaskInfo
from stage_map.object.utils import Point
from stage_map.object.player import Player
from stage_map.object.ball import Ball


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

	map_ready_flag: bool = False
	map_read_only_flag: bool = True
	map_name: str = "아직 불러오지 않음"
	map_theme: ThemeInfo = ThemeInfo("default")
	map_path: Path
	map_width: int = 20
	map_height: int = 12
	map_file_name: str
	background_image: np.ndarray = np.zeros((800, 1100, 3), dtype=np.uint8)
	mask_info: MaskInfo = MaskInfo(np.zeros((800, 1100, 1), dtype=np.uint8))
	player_info: dict
	ball_data_list: list[dict] = []
	# _checkpoint_grid: list[list[int]] = [[0] * (map_width + 2) for _ in range(map_width + 2)]  # padding
	_checkpoint_grid: np.array = np.zeros((map_height + 2, map_width + 2), dtype=np.uint8)
	_checkpoint_count: int = 0
	_checkpoint_respon_pos: list[Point] = []
	_goal_num: int = 0

	def __init__(self, map_file_path: Path, map_read_only_flag: bool = True):
		self.map_file_path = map_file_path
		self.map_file_name = map_file_path.name

		self.load_map(map_file_path)

	def load_map(self, file_path: Path):
		with open(file_path, 'r') as file:
			map_height = self.map_height
			map_width = self.map_width

			data = json.load(file)
			if "map_name" not in data:
				raise KeyError(f"{Path}에 필수 키 \"map_name\"이(가) 존재하지 않습니다.")
			self.map_name = data["map_name"]
			if "map_theme" in data:
				self.map_theme = self.ThemeInfo(data["map_theme"])
			if "matrix" not in data:
				raise KeyError(f"{Path}에 필수 키 \"matrix\"이(가) 존재하지 않습니다.")
			map_matrix = np.array(data["matrix"])
			if map_matrix.shape != (self.map_height, self.map_width):
				raise ValueError("맵 크기가 12x20이 아닙니다.")
			padded_matrix = np.pad(map_matrix, ((1, 0), (1, 0)), mode='constant', constant_values=0)
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

			# 바닥 타일 배치
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
					else:
						raise ValueError(f"알 수 없는 타일 값: {padded_matrix[y][x]}")

			# 체크포인트 지역 분할
			checkpoint_grid_temp = np.zeros((map_height, map_width), dtype=np.uint8)
			def checkpoint_bfs(i, j):
				# if self._checkpoint_grid[i, j] == 0:
				if padded_matrix[i, j] == 0:
					pass
			for y in range(1, map_height + 1):
				for x in range(1, map_width + 1):
					pass

			# 벽(경계) 추가
			for y in range(0, map_height):
				for x in range(0, map_width):
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

			if "player_info" in data:
				self.player_info = data["player_info"]
			if "ball_list" in data:
				self.ball_data_list = data["ball_list"]
			self.map_ready_flag = True

	def get_player(self):
		return Player(self.player_info)

	def get_ball_data(self):
		ball_object_list = []
		for data in self.ball_data_list:
			ball_object_list.append(Ball(data))
		return ball_object_list

	@property
	def checkpoint_grid(self):
		return self._checkpoint_grid[1:-1, 1:-1]


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
