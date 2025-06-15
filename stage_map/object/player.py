import numpy as np

from stage_map.object.utils import Point, Object, EnvData, ObjectManager
from stage_map.mask import MaskInfo


class Player(Object):
	size = (36, 36)
	die_reward = -1
	wall_reward = -0.02
	wall_die = True
	# move_reward = -0.0005
	move_reward = -0.0
	checkpoint_reward = 1
	coin_reward = 3
	# coin_smell_reward = 0.002
	coin_smell_reward = 0.0
	# ball_smell_reward = -0.1
	goal_reward = 5
	# change_action_penalty = -0.001
	change_action_penalty = 0
	# step_penalty = -0.0005  # 최종 보상에 적용되는 패널티
	step_penalty = 0  # 최종 보상에 적용되는 패널티
	# stop_penalty = -0.05
	# env_end_penalty = -100

	def __init__(
		self,
		object_manager: ObjectManager,
		player_id: int,
		data: dict,
		random_init_pos: bool = True,
	):
		self.init_pos = Point(*data["init_pos"])
		self.init_pos_list = [Point(*data["init_pos"])]
		if random_init_pos and "extra_pos" in data:
			self.init_pos_list.extend(Point(*pos) for pos in data["extra_pos"])
		super().__init__(
			object_manager=object_manager,
			pos=np.random.choice(self.init_pos_list).copy(),
			anchor=Point(0.5, 0.5),
			priority=100,
		)
		self.player_id = player_id
		self.speed = 6
		self.action = 0
		self.previous_action = 0
		self.is_dead = False
		self.is_goal = False
		self.step_count = 0
		self.checkpoint_id_set = set()
		self.coin_id_set = set()
		self.previous_coin_smell_reward = 0
		self._reward = 0

		s = self.speed
		self.movement = [
			(0, 0),  # action 0: 정지
			(0, -s),  # action 1: 위
			(s, -s),  # action 2: 오른쪽 위
			(s, 0),  # action 3: 오른쪽
			(s, s),  # action 4: 오른쪽 아래
			(0, s),  # action 5: 아래
			(-s, s),  # action 6: 왼쪽 아래
			(-s, 0),  # action 7: 왼쪽
			(-s, -s)  # action 8: 왼쪽 위
		]

	# 보상
	@property
	def reward(self):
		return self._reward

	def collision_mask(self, env_data: EnvData) -> None:
		pass

	def compute_inertia_penalty(
		self,
		penalty_weight: float = -1.0,
		norm_type: str = "l2",  # "l1" 또는 "l2"
	) -> float:
		prev_vec = np.array(self.movement[self.previous_action])
		curr_vec = np.array(self.movement[self.action])
		delta_v = curr_vec - prev_vec
		if norm_type == "l1":
			norm = np.sum(np.abs(delta_v))
		elif norm_type == "l2":
			norm = np.linalg.norm(delta_v)
		else:
			raise ValueError("Invalid norm_type. Use 'l1' or 'l2'.")
		return penalty_weight * norm

	def inverse_square_reward(self, distance):
		"""
		Inverse-Square 보상 함수

		Args:
		    distance (float or np.ndarray): 코인까지의 거리
		    a (float): 최대 보상 (거리 0일 때 보상)
		    r (float): 거리 스케일 파라미터 (값이 클수록 천천히 감소)

		Returns:
		    float or np.ndarray: 거리별 보상 값
		"""
		return self.coin_smell_reward / (1 + (distance / 200) ** 2)

	def reset(self, env_data: EnvData, seed=None):
		rng = np.random.default_rng(seed=seed)
		self.pos = rng.choice(self.init_pos_list).copy()
		self.action = 0
		self.previous_action = 0
		self.is_dead = False
		self.is_goal = False
		self.step_count = 0
		self.checkpoint_id_set = set()
		self.coin_id_set = set()
		self.previous_coin_smell_reward = 0
		env_data.player_trail_canvas[self.player_id, ...] = 0

	def step(self, env_data: EnvData) -> None:
		mask = env_data.dynamic_collision_mask
		checkpoint_mask = env_data.checkpoint_id_mask
		w, h = self.size
		a_x, a_y = int(self.pos.x), int(self.pos.y)
		x1 = int(a_x - self.anchor.x * w)
		y1 = int(a_y - self.anchor.y * h)
		x2 = x1 + w
		y2 = y1 + h
		self._reward = 0

		# dx, dy 값 업데이트
		dx, dy = self.movement[self.action]

		# # 이동(관성) 패널티
		# self._reward += self.compute_inertia_penalty(penalty_weight=self.change_action_penalty, norm_type="l2")

		# 지속 정지 패널티

		# 이동 벽 검사
		wall_flag = False
		if dy < 0 and (mask[y1+dy, a_x-1:a_x+1] & MaskInfo.MaskLayer.WALL).any():  # 위쪽
			dy = 0
			wall_flag = True
		if 0 < dy and (mask[y1 + h + dy, a_x - 1:a_x + 1] & MaskInfo.MaskLayer.WALL).any():  # 아래쪽
			dy = 0
			wall_flag = True
		if dx < 0 and (mask[a_y-1:a_y+1, x1+dx] & MaskInfo.MaskLayer.WALL).any():  # 왼쪽
			dx = 0
			wall_flag = True
		if 0 < dx and (mask[a_y - 1:a_y + 1, x1 + w + dx] & MaskInfo.MaskLayer.WALL).any():  # 오른쪽
			dx = 0
			wall_flag = True
		if wall_flag:
			if self.wall_die:
				self.is_dead = True
				self._reward += self.die_reward
				print(f"env: {env_data.name}, Player {self.player_id} is wall die")
			else:
				self._reward += self.wall_reward

		# 제자리 벽 검사
		# 위쪽
		if (mask[y1, a_x-1:a_x+1] & MaskInfo.MaskLayer.WALL).any():
			dy += self.speed
		# 아래쪽
		if (mask[y1+h, a_x-1:a_x+1] & MaskInfo.MaskLayer.WALL).any():
			dy -= self.speed
		# 왼쪽
		if (mask[a_y-1:a_y+1, x1] & MaskInfo.MaskLayer.WALL).any():
			dx += self.speed
		# 오른쪽
		if (mask[a_y-1:a_y+1, x1+w] & MaskInfo.MaskLayer.WALL).any():
			dx -= self.speed

		# 플레이어 이동
		self.pos += Point(dx, dy)

		# 플레이어 위치 업데이트
		a_x, a_y = int(self.pos.x), int(self.pos.y)
		x1 = int(a_x - self.anchor.x * w)
		y1 = int(a_y - self.anchor.y * h)
		x2 = x1 + w
		y2 = y1 + h

		# 체크포인트 지역 검사
		# if (mask[y1:y1+h, x1:x1+w] & MaskInfo.MaskLayer.CHECKPOINT_ZONE).any():
		checkpoint_id = checkpoint_mask[y1:y1+h, x1:x1+w].max()
		if 0 < checkpoint_id and checkpoint_id not in self.checkpoint_id_set:
			name = env_data.name
			if env_data.checkpoint_count - 1 <= len(self.checkpoint_id_set):  # 마지막 체크포인트일 경우
				if len(self.coin_id_set) == env_data.coin_count:  # 코인을 모두 획득한 경우
					self._reward += max(self.goal_reward + self.step_penalty * self.step_count, self.goal_reward // 10)
					self.is_goal = True
				else:  # 코인을 덜 획득한 경우
					self._reward += (
						self.goal_reward * (len(self.coin_id_set) / env_data.coin_count)
						+ self.step_penalty * self.step_count
					)
					self.is_goal = True
				print(
					f"env: {name}, Player {self.player_id} reached the final checkpoint at step {self.step_count}."
					f"({len(self.coin_id_set)} / {env_data.coin_count}, reward: {self._reward})"
				)
			else:  # 일반 체크포인트인 경우
				if self.step_count == 0:  # 첫 번째 체크포인트는 보상 없음
					self.checkpoint_id_set.add(checkpoint_id)
				else:
					self._reward += self.checkpoint_reward
					self.checkpoint_id_set.add(checkpoint_id)
					print(f"env: {name}, Player {self.player_id} reached a checkpoint at step {self.step_count}.")

		# 코인 획득 검사
		coin_object_list: list = env_data.coin_object_list
		coin_smell_max_reward = 0
		for coin_i, coin_obj in enumerate(coin_object_list):
			if coin_obj.id in self.coin_id_set:
				continue  # 이미 획득한 코인은 무시
			closest_x = max(x1, min(int(coin_obj.pos.x), x2))
			closest_y = max(y1, min(int(coin_obj.pos.y), y2))
			diff_x = closest_x - int(coin_obj.pos.x)
			diff_y = closest_y - int(coin_obj.pos.y)
			coin_radius = 13  # 코인의 반지름
			center_to_edge_distance = np.sqrt(diff_x ** 2 + diff_y ** 2)
			coin_smell_reward = self.inverse_square_reward(center_to_edge_distance - coin_radius)
			if coin_smell_max_reward < coin_smell_reward:
				coin_smell_max_reward = coin_smell_reward
			if center_to_edge_distance < coin_radius:  # 플레이어와 코인의 거리 비교
				self.coin_id_set.add(coin_obj.id)
				self._reward += self.coin_reward - self.coin_smell_reward * self.step_count
				print(f"env: {env_data.name}, Player {self.player_id} collected coin {coin_obj.id} at step {self.step_count}.")

		if self.previous_coin_smell_reward < coin_smell_max_reward:
			self._reward += coin_smell_max_reward
		self.previous_coin_smell_reward = coin_smell_max_reward

		# 죽음 검사
		if (mask[y1:y1+h, x1:x1+w] & MaskInfo.MaskLayer.BALL).any():
			self.is_dead = True
			self._reward += self.die_reward

		self._reward += self.move_reward * (dx**2 + dy**2)**0.5 / self.speed


		self.previous_action = self.action
		self.step_count += 1

	def draw(self, env_data: EnvData, view_canvas: np.ndarray | None = None) -> None:
		if view_canvas is not None:
			canvas = view_canvas
		else:
			canvas = env_data.dynamic_canvas.copy()
		mask = env_data.dynamic_collision_mask

		w, h = self.size
		x1 = int(self.pos.x - self.anchor.x * w)
		y1 = int(self.pos.y - self.anchor.y * h)
		x2 = x1 + w
		y2 = y1 + h
		canvas[y1:y2, x1:x2] = (0, 0, 0)
		if view_canvas is None and env_data.player_trail_canvas is not None:
			roi = env_data.player_trail_canvas[self.player_id, y1:y2, x1:x2]
			# mask = roi < 255
			# roi[mask] += 32
			roi[...] = np.minimum(roi.astype(np.int16) + 32, 255).astype(np.uint8)
		x1 += 6
		y1 += 6
		x2 -= 6
		y2 -= 6
		canvas[y1:y2, x1:x2] = (255, 0, 0)
		# mask[y1:y2, x1:x2] = MaskInfo.MaskLayer.PLAYER
		if view_canvas is None:
			env_data.player_draw_canvas[self.player_id, ...] = canvas