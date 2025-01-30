import cv2

from stage_map.mask import MaskInfo
from stage_map.object.utils import Point, Object, EnvData


class Coin(Object):
	mask_layer_id: int = MaskInfo.MaskLayer.COIN

	def __init__(self, data: dict, priority: int = 0):
		super().__init__(pos=Point(*data["path"][0]), priority=priority)

	def collision_mask(self, env_data: EnvData) -> None:
		x1 = int(self.pos.x) - 13
		y1 = int(self.pos.y) - 13
		x2 = int(self.pos.x) + 13
		y2 = int(self.pos.y) + 13
		env_data.collision_mask[y1:y2, x1:x2] |= self.mask_layer_id

	def step(self, env_data: EnvData) -> None:
		# 환경 데이터
		fps = env_data.frame_per_second
		cp = env_data.current_frame

		# 거리 비율 계산 (현재 진행된 거리의 비율)
		path_distance_ratio = (cp % (fps * self.path_time)) / (fps * self.path_time)

		# 경로 상에서의 현재 위치 계산
		i, d, cum_dis = 0, 0, 0
		target_distance = self.path_distance_sum * path_distance_ratio  # 경로 상의 거리 기준 진행 위치
		for i, d in enumerate(self.path_distance_list):
			if target_distance <= cum_dis + d:
				break
			cum_dis += d

		segment_ratio = (target_distance - cum_dis) / d  # 현재 세그먼트 내에서 비율 계산
		self.pos = self.path[i].interpolate(self.path[i + 1], segment_ratio)

	def draw(self, env_data: EnvData) -> None:
		shift: int = 10
		pos = self.pos
		pos -= .5
		cv2.circle(
			img=env_data.canvas,
			center=(pos * (2.0 ** shift)).to_tuple_int(),
			radius=int(12.5 * (2 ** shift)),
			color=(0, 0, 0),
			thickness=-1,
			lineType=cv2.LINE_AA,
			shift=shift,
		)
		cv2.circle(
			img=env_data.canvas,
			center=(pos * (2.0 ** shift)).to_tuple_int(),
			radius=int(6.5 * (2 ** shift)),
			color=(0, 0, 255),
			thickness=-1,
			lineType=cv2.LINE_AA,
			shift=shift,
		)
