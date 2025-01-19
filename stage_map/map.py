from pathlib import Path
import numpy as np
import json
import cv2

from stage_map.object.utils import Point
from stage_map.object.pleyer import Player
from stage_map.object.ball import Ball


class MapData:
	class ThemeInfo:
		def __init__(self, theme_name: str):
			if theme_name == "default" or theme_name == "blue":
				self.letterbox_color = (0, 0, 0)
				self.background_color = (180, 181, 254)
				self.floor_tile_color_list = [(247, 247, 255), (230, 230, 255)]
				self.ball_tile_color = (0, 0, 255)
				self.safe_zone_color = (182, 254, 180)
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
	map_file_name: str
	current_frame: int = 0
	frame_per_second: int = 30
	background_image: np.ndarray = np.zeros((800, 1100, 3), dtype=np.uint8)
	solid_image: np.ndarray = np.zeros((800, 1100, 1), dtype=np.uint8)
	player_init_pos: Point
	ball_data_list: list[dict] = []

	def __init__(self, map_file_path: Path, map_read_only_flag: bool = True):
		self.map_file_path = map_file_path
		self.map_file_name = map_file_path.name

	def load_map(self, file_path: Path):
		with open(file_path, 'r') as file:
			data = json.load(file)
			if "map_name" in data:
				self.map_name = data["map_name"]
			if "map_theme" in data:
				self.map_theme = self.ThemeInfo(data["map_theme"])
			if "matrix" in data:
				map_matrix = np.array(data["matrix"])
				if map_matrix.shape != (12, 20):
					raise ValueError("맵 크기가 12x20이 아닙니다.")
				padded_matrix = np.pad(map_matrix, ((1, 0), (1, 0)), mode='constant', constant_values=0)
				b_img = self.background_image
				s_img = self.solid_image
				b_img[:53,:] = self.map_theme.letterbox_color
				b_img[53:-53, :] = self.map_theme.background_color
				s_img[53:-53, :] = True
				b_img[-53:, :] = self.map_theme.letterbox_color
				grid_size = 50
				for y in range(1, 13):
					for x in range(1, 21):
						if padded_matrix[y][x] == 0:
							continue
						elif padded_matrix[y][x] == 1:
							tile_color = self.map_theme.floor_tile_color_list[(x + y) % 2]
						elif padded_matrix[y][x] == 2:
							tile_color = self.map_theme.safe_zone_color
						else:
							raise ValueError(f"알 수 없는 타일 값: {padded_matrix[y][x]}")
						y1 = 53 + y * grid_size
						x1 = x * grid_size
						b_img[y1:y1 + grid_size, x1:x1 + grid_size] = tile_color
			if "player_init_pos" in data:
				self.player_init_pos = Point(*data["player_init_pos"])
			if "ball_list" in data:
				self.ball_data_list = data["ball_list"]

	def get_player(self):
		return Player(self.player_init_pos, anchor=Point(0.5, 0.5), priority=100)

	def get_ball_data(self):
		ball_object_list = []
		for data in self.ball_data_list:
			ball_object_list.append(Ball(Point(*data["pos"])))
		return ball_object_list


if __name__ == "__main__":
	map_data = MapData(Path(__file__).parent / "map_file" / "map1.json")
	map_data.load_map(map_data.map_file_path)
	print(map_data.map_name)
	print(map_data.map_theme)
	# RGA to BGR
	map_data.background_image = cv2.cvtColor(map_data.background_image, cv2.COLOR_RGB2BGR)
	cv2.imshow("Map", map_data.background_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()