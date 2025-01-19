from .utils import Point, Object, EnvData


class Ball(Object):

	def __init__(self, data: dict, priority: int = 0):
		super().__init__(pos=Point(*data["path"][0]), priority=priority)
		self.image = None
		self.collision_mask = None

	def collision_mask(self, env_data: EnvData) -> None:
		pass

	def step(self, env_data: EnvData) -> None:
		pass

	def draw(self, env_data: EnvData) -> None:
		pass