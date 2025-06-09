import cv2
import numpy as np

from stage_map.mask import MaskInfo
from stage_map.object.utils import Point, Object, EnvData, ObjectManager
# from stage_map.object.player import Player


class Coin(Object):
	mask_layer_id: int = MaskInfo.MaskLayer.COIN

	def __init__(
		self,
		object_manager: ObjectManager,
		data: dict,
		priority: int = 200,
	):
		super().__init__(
			object_manager=object_manager,
			pos=Point(*data["pos"]),
			priority=priority,
		)
		self.first_step_flag = True

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
		pass
		# # 플레이어가 코인을 획득했는지 확인
		# if self.first_step_flag:
		# 	self.first_step_flag = False
		# 	return
		# player_object_list: list = env_data.player_object_list
		# for player_id, player_obj in enumerate(player_object_list):
		# 	if self.id in player_obj.coin_id_set:
		# 		continue
		# 	player_w, player_h = player_obj.size
		# 	player_x1 = int(player_obj.pos.x - player_obj.anchor.x * player_w)
		# 	player_y1 = int(player_obj.pos.y - player_obj.anchor.y * player_h)
		# 	player_x2 = player_x1 + player_w
		# 	player_y2 = player_y1 + player_h
		# 	closest_x = max(player_x1, min(int(self.pos.x), player_x2))
		# 	closest_y = max(player_y1, min(int(self.pos.y), player_y2))
		# 	dx = closest_x - int(self.pos.x)
		# 	dy = closest_y - int(self.pos.y)
		# 	center_to_edge_distance = np.sqrt(dx ** 2 + dy ** 2)
		# 	if center_to_edge_distance < 13:  # 플레이어와 코인의 거리 비교
		# 		player_obj.coin_id_set.add(self.id)

	def draw(self, env_data: EnvData) -> None:
		shift: int = 10
		pos = self.pos
		pos -= .5
		player_object_list: list = env_data.player_object_list
		canvas = env_data.player_draw_canvas
		for player_id, player_obj in enumerate(player_object_list):
			if self.id in player_obj.coin_id_set:
				continue  # 이미 획득한 코인은 그리지 않음
			cv2.circle(
				img=canvas[player_id],
				center=(pos * (2.0 ** shift)).to_tuple_int(),
				radius=int(12.5 * (2 ** shift)),
				color=(0, 0, 0),
				thickness=-1,
				lineType=cv2.LINE_AA,
				shift=shift,
			)
			cv2.circle(
				img=canvas[player_id],
				center=(pos * (2.0 ** shift)).to_tuple_int(),
				radius=int(6.5 * (2 ** shift)),
				color=(255, 255, 0),
				thickness=-1,
				lineType=cv2.LINE_AA,
				shift=shift,
			)

	def draw_rendering(self, env_data: EnvData, view_canvas: np.ndarray) -> None:
		"""
		렌더링용 드로우 함수.
		이 함수는 주로 시각화나 디버깅 목적으로 사용됩니다.
		:param env_data: 환경 데이터
		:param view_canvas: 뷰 캔버스 (렌더링용)
		:return:
		"""
		shift: int = 10
		pos = self.pos
		pos -= .5
		cv2.circle(
			img=view_canvas,
			center=(pos * (2.0 ** shift)).to_tuple_int(),
			radius=int(12.5 * (2 ** shift)),
			color=(0, 0, 0),
			thickness=-1,
			lineType=cv2.LINE_AA,
			shift=shift,
		)
		cv2.circle(
			img=view_canvas,
			center=(pos * (2.0 ** shift)).to_tuple_int(),
			radius=int(6.5 * (2 ** shift)),
			color=(255, 255, 0),
			thickness=-1,
			lineType=cv2.LINE_AA,
			shift=shift,
		)