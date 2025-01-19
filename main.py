from pathlib import Path
import numpy as np
import cv2
import pygame
import gymnasium as gym

from stage_map.map import MapData
from stage_map import env

if __name__ == "__main__":
	map_data = MapData(Path(__file__).parent / "stage_map" / "map_file" / "map1.json")
	map_data.load_map(map_data.map_file_path)
	print(map_data.map_name)
	print(map_data.map_theme)
	# RGA to BGR
	map_data.background_image = cv2.cvtColor(map_data.background_image, cv2.COLOR_RGB2BGR)
	cv2.imshow("Map", map_data.background_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

