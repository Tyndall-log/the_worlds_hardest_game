from typing import ClassVar
import threading
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


class Point:
	x: float
	y: float

	def __init__(self, x: float | int = 0, y: float | int = 0):
		if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
			raise TypeError("x와 y는 int 또는 float 타입이어야 합니다.")
		self.x = float(x)
		self.y = float(y)

	def __add__(self, other: "Point | int | float") -> "Point":
		if isinstance(other, Point):
			return Point(self.x + other.x, self.y + other.y)
		elif isinstance(other, (int, float)):
			return Point(self.x + other, self.y + other)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __sub__(self, other: "Point | int | float") -> "Point":
		if isinstance(other, Point):
			return Point(self.x - other.x, self.y - other.y)
		elif isinstance(other, (int, float)):
			return Point(self.x - other, self.y - other)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __mul__(self, other: "Point | int | float") -> "Point":
		if isinstance(other, Point):
			return Point(self.x * other.x, self.y * other.y)
		elif isinstance(other, (int, float)):
			return Point(self.x * other, self.y * other)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __truediv__(self, other: "Point | int | float") -> "Point":
		if isinstance(other, Point):
			return Point(self.x / other.x, self.y / other.y)
		elif isinstance(other, (int, float)):
			return Point(self.x / other, self.y / other)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __floordiv__(self, other: "Point | int | float") -> "Point":
		if isinstance(other, Point):
			return Point(self.x // other.x, self.y // other.y)
		elif isinstance(other, (int, float)):
			return Point(self.x // other, self.y // other)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __radd__(self, other: "int | float") -> "Point":
		if isinstance(other, (int, float)):
			return Point(other + self.x, other + self.y)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __rsub__(self, other: "int | float") -> "Point":
		if isinstance(other, (int, float)):
			return Point(other - self.x, other - self.y)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __rmul__(self, other: "int | float") -> "Point":
		if isinstance(other, (int, float)):
			return Point(other * self.x, other * self.y)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __rtruediv__(self, other: "int | float") -> "Point":
		if isinstance(other, (int, float)):
			return Point(other / self.x, other / self.y)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def __rfloordiv__(self, other: "int | float") -> "Point":
		if isinstance(other, (int, float)):
			return Point(other // self.x, other // self.y)
		else:
			raise TypeError(f"지원하지 않는 타입: {type(other)}")

	def magnitude(self) -> float:
		return math.sqrt(self.x ** 2 + self.y ** 2)

	def distance_to(self, other: "Point") -> float:
		if not isinstance(other, Point):
			raise TypeError(f"distance_to는 Point 객체만 지원합니다. 제공된 타입: {type(other)}")
		if other.x is None or other.y is None:
			raise ValueError("다른 Point 객체의 좌표가 유효하지 않습니다.")
		return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

	def __neg__(self) -> "Point":
		return Point(-self.x, -self.y)

	def __abs__(self) -> "Point":
		return Point(abs(self.x), abs(self.y))

	def __eq__(self, other: object) -> bool:
		if isinstance(other, Point):
			return self.x == other.x and self.y == other.y
		return False

	def __hash__(self) -> int:
		return hash((self.x, self.y))

	def __str__(self) -> str:
		return f"({self.x}, {self.y})"

	def __repr__(self) -> str:
		return f"Point(x={self.x:.2f}, y={self.y:.2f})"


class EnvData:
	canvas: np.ndarray  # 캔버스
	collision_mask: np.ndarray  # 충돌 마스크

	def __init__(self, canvas: np.ndarray, collision_mask: np.ndarray):
		self.canvas = canvas
		self.collision_mask = collision_mask


class ObjectManager:
	__id: ClassVar[int] = 0
	__lock: ClassVar[threading.Lock] = threading.Lock()  # 스레드 안전성
	__object_list: ClassVar[list["Object"]] = []

	@classmethod
	def get_new_id(cls, obj: "Object") -> int:
		"""새로운 ID를 생성하고 반환합니다."""
		with cls.__lock:  # 동시 접근 방지
			cls.__id += 1
			cls.__object_list.append(obj)
			return cls.__id


class Object(ABC):
	pos: Point  # 오브젝트의 위치
	anchor: Point  # 오브젝트의 앵커 위치
	priority: int  # 낮을수록 먼저 계산되고, 먼저 그려짐
	image: np.ndarray  # 이미지
	collision_mask: np.ndarray  # 마스크
	_id: int

	def __init__(
		self,
		pos: Point,
		anchor: Point = Point(0, 0),
		priority: int = 0
	):
		"""
		:param pos: 오브젝트의 위치
		:param anchor: 오브젝트의 앵커 위치 (0.0 ~ 1.0)
		:param priority: 오브젝트의 우선순위 (낮을수록 먼저 계산되고, 먼저 그려짐)
		"""
		self.pos = pos
		self.priority = priority
		self.anchor = anchor
		self._id = ObjectManager.get_new_id(self)

	@property
	def id(self) -> int:
		"""객체의 고유 ID"""
		return self._id

	@abstractmethod
	def collision_mask(self, env_data: EnvData) -> np.ndarray:
		"""
		매 프레임마다 오브젝트의 충돌 마스크를 업데이트합니다.
		:return:
		"""
		pass

	@abstractmethod
	def step(self, env_data: EnvData) -> None:
		"""
		매 프레임마다 오브젝트의 상태를 업데이트합니다.
		:return:
		"""
		pass

	@abstractmethod
	def draw(self, env_data: EnvData) -> None:
		"""
		매 프레임마다 오브젝트를 그립니다.
		:param canvas:
		:return:
		"""
		pass

	def __str__(self):
		class_name = type(self).__name__
		return f"{class_name}(id={self.id}, pos={self.pos})"


