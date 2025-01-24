from stage_map.object.utils import Point, Object, EnvData
from stage_map.mask import MaskInfo


class Player(Object):

	def __init__(self, data: dict):
		super().__init__(pos=Point(*data["init_pos"]), anchor=Point(0.5, 0.5), priority=100)
		self.init_pos = Point(*data["init_pos"])
		self.speed = 6
		self.action = 0
		self.size = (36, 36)
		self.die_flag = False
		self._reward = 0
		self.die_reward = -10
		self.wall_reward = -0.1
		self.move_reward = -0.001
		self.checkpoint_reward = 10
		self.goal_reward = 100

		s = self.speed
		self.movement = [
			(0, 0),  # action 0: 정지 (사용하지 않음)
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

	def step(self, env_data: EnvData) -> None:
		mask = env_data.collision_mask
		w, h = self.size
		a_x, a_y = int(self.pos.x), int(self.pos.y)
		x1 = int(a_x - self.anchor.x * w)
		y1 = int(a_y - self.anchor.y * h)
		self._reward = 0

		# dx, dy 값 업데이트
		dx, dy = self.movement[self.action]

		# 이동 벽 검사
		if dy < 0 and (mask[y1+dy, a_x-1:a_x+1] & MaskInfo.MaskLayer.WALL).any():  # 위쪽
			dy = 0
			self._reward += self.wall_reward
		if 0 < dy and (mask[y1 + h + dy, a_x - 1:a_x + 1] & MaskInfo.MaskLayer.WALL).any():  # 아래쪽
			dy = 0
			self._reward += self.wall_reward
		if dx < 0 and (mask[a_y-1:a_y+1, x1+dx] & MaskInfo.MaskLayer.WALL).any():  # 왼쪽
			dx = 0
			self._reward += self.wall_reward
		if 0 < dx and (mask[a_y - 1:a_y + 1, x1 + w + dx] & MaskInfo.MaskLayer.WALL).any():  # 오른쪽
			dx = 0
			self._reward += self.wall_reward

		# 제자리 벽 검사
		# 위쪽
		if (mask[y1, a_x-1:a_x+1] & MaskInfo.MaskLayer.WALL).any():
			dy += 6
		# 아래쪽
		if (mask[y1+h, a_x-1:a_x+1] & MaskInfo.MaskLayer.WALL).any():
			dy -= 6
		# 왼쪽
		if (mask[a_y-1:a_y+1, x1] & MaskInfo.MaskLayer.WALL).any():
			dx += 6
		# 오른쪽
		if (mask[a_y-1:a_y+1, x1+w] & MaskInfo.MaskLayer.WALL).any():
			dx -= 6

		if 0 < dx:
			self._reward += self.move_reward
		if 0 < dy:
			self._reward += self.move_reward

		self.pos += Point(dx, dy)



	def draw(self, env_data: EnvData) -> None:
		canvas = env_data.canvas
		mask = env_data.collision_mask

		w, h = self.size
		x1 = int(self.pos.x - self.anchor.x * w)
		y1 = int(self.pos.y - self.anchor.y * h)
		x2 = x1 + w
		y2 = y1 + h
		canvas[y1:y2, x1:x2] = (0, 0, 0)
		x1 += 6
		y1 += 6
		x2 -= 6
		y2 -= 6
		canvas[y1:y2, x1:x2] = (255, 0, 0)
		# mask[y1:y2, x1:x2] = MaskInfo.MaskLayer.PLAYER
		self.action = 0
