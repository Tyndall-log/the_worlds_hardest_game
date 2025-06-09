import cv2
import numpy as np

from stage_map.mask import MaskInfo
from stage_map.object.utils import Point, Object, EnvData, ObjectManager


class Ball(Object):
	path: list[Point]
	path_distance_list: list[float]
	path_distance_sum: float
	path_time: float
	mask_layer_id: int = MaskInfo.MaskLayer.BALL

	def __init__(self,
		object_manager: ObjectManager,
		data: dict,
		priority: int = 0
	):
		super().__init__(
			object_manager=object_manager,
			pos=Point(*data["path"][0]),
			priority=priority
		)
		self.path = [Point(*pos) for pos in data["path"]]
		self.path_distance_list = [self.path[i].distance_to(self.path[i + 1]) for i in range(len(self.path) - 1)]
		self.path_distance_sum = sum(self.path_distance_list)
		self.path_time = data["time"]
		if "path_mode" in data:
			mode = data["path_mode"]
			if mode == "standard":
				pass
			elif mode == "bounce":
				self.path += self.path[:-1][::-1]
				self.path_distance_list += self.path_distance_list[::-1]
				self.path_distance_sum *= 2
				self.path_time *= 2
			else:
				raise ValueError(f"\"path_mode\"에 알 수 없는 모드입니다.{mode}")
		# TODO: 공을 그룹화 하고 회전할 수 있는 옵션 추가(그리드 배치 및 회전각(내부 라인 지원) 배치 지원)

	def collision_mask(self, env_data: EnvData) -> None:
		x1 = int(self.pos.x) - 13
		y1 = int(self.pos.y) - 13
		x2 = int(self.pos.x) + 13
		y2 = int(self.pos.y) + 13

		# 26x26 원형 마스크 생성
		size = 26
		yy, xx = np.ogrid[:size, :size]
		cx, cy = 12.5, 12.5
		circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 12 ** 2  # 반지름 13짜리 원

		# 대상 영역 슬라이스
		target = env_data.dynamic_collision_mask[y1:y2, x1:x2]

		# 원 모양 영역에만 OR 연산 적용
		target[circle_mask] |= self.mask_layer_id

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

		if d == 0:
			# 현재 세그먼트가 없으면 마지막 위치로 설정
			self.pos = self.path[-1]
		else:
			segment_ratio = (target_distance - cum_dis) / d  # 현재 세그먼트 내에서 비율 계산
			self.pos = self.path[i].interpolate(self.path[i + 1], segment_ratio)

		# 충돌 마스크 업데이트
		self.collision_mask(env_data)

	def draw(self, env_data: EnvData) -> None:
		shift: int = 10
		pos = self.pos
		pos -= .5
		cv2.circle(
			img=env_data.dynamic_canvas,
			center=(pos * (2.0 ** shift)).to_tuple_int(),
			radius=int(12.5 * (2 ** shift)),
			color=(0, 0, 0),
			thickness=-1,
			lineType=cv2.LINE_AA,
			shift=shift,
		)
		cv2.circle(
			img=env_data.dynamic_canvas,
			center=(pos * (2.0 ** shift)).to_tuple_int(),
			radius=int(6.5 * (2 ** shift)),
			color=(0, 0, 255),
			thickness=-1,
			lineType=cv2.LINE_AA,
			shift=shift,
		)
