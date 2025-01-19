from .utils import Point, Object, EnvData


class Player(Object):

	def collision_mask(self, env_data: EnvData) -> None:
		pass

	def step(self, env_data: EnvData) -> None:
		pass

	def draw(self, env_data: EnvData) -> None:
		pass
