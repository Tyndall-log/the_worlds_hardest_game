from __future__ import annotations
# from typing import ClassVar
import bisect
import threading
from abc import ABC, abstractmethod
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np


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

	def interpolate(self, other, ratio):
		"""두 점 사이를 비율(ratio)로 보간"""
		return Point(
			self.x + (other.x - self.x) * ratio,
			self.y + (other.y - self.y) * ratio,
		)

	def to_tuple(self) -> tuple[float, float]:
		return self.x, self.y

	def to_tuple_int(self) -> tuple[int, int]:
		return int(self.x), int(self.y)

	@classmethod
	def from_tuple(cls, coords: tuple[float, float]) -> Point:
		if not isinstance(coords, tuple) or len(coords) != 2:
			raise TypeError("coords는 (x, y) 형태의 튜플이어야 합니다.")
		return cls(float(coords[0]), float(coords[1]))

	def copy(self) -> Point:
		"""Point 객체를 복사하여 반환합니다."""
		return Point(self.x, self.y)

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
	def __init__(
		self,
		name: str,
		object_manager: ObjectManager,
		player_object_list: list[Object],
		coin_object_list: list[Object],
		# canvas: np.ndarray,
		# collision_mask: np.ndarray,
		# checkpoint_id_mask: np.ndarray,
		# checkpoint_count: int,
		map_data,
		frame_per_second: int = 30,
		current_frame: int = 0
	):
		"""
		환경 데이터를 초기화합니다.
		:param object_manager: 오브젝트 관리자
		:param player_object_list: 플레이어 오브젝트 리스트
		:param coin_object_list: 코인 오브젝트 리스트
		:param map_data: 맵 데이터
		:param frame_per_second: 초당 프레임 수
		:param current_frame: 현재 프레임
		"""
		canvas = map_data.background_image.copy()
		collision_mask = map_data.mask_info.mask_image.copy()
		checkpoint_id_mask = map_data.checkpoint_id_mask.copy()
		checkpoint_count = map_data.checkpoint_count

		self.name = name  # 환경 이름
		self.object_manager = object_manager  # 오브젝트 관리자
		self.player_num = len(player_object_list)  # 플레이어 수
		self.player_object_list: list[Object] = player_object_list  # 플레이어 오브젝트 리스트
		self.coin_object_list: list[Object] = coin_object_list
		self.init_canvas = canvas  # 캔버스
		self.dynamic_canvas = canvas.copy()  # 동적 캔버스
		self.player_draw_canvas = np.zeros((self.player_num, *canvas.shape), dtype=canvas.dtype)  # 플레이어별 그리기용 캔버스
		self.player_trail_canvas = np.zeros((self.player_num, *canvas.shape[:-1]), dtype=canvas.dtype)  # 플레이어 트레일 캔버스
		self.static_collision_mask = collision_mask  # 충돌 마스크
		self.dynamic_collision_mask = collision_mask.copy()  # 동적 충돌 마스크
		self.checkpoint_id_mask = checkpoint_id_mask  # 체크포인트 ID 마스크
		self.checkpoint_count = checkpoint_count  # 체크포인트 개수
		self.coin_count = map_data.coin_count  # 코인 개수
		# self.individual_collision_mask = np.zeros_like(collision_mask, dtype=np.uint8)  # 개별 충돌 마스크
		self.frame_per_second = frame_per_second
		self.current_frame = current_frame

	def reset(
		self,
		canvas: np.ndarray | None = None,
		frame_per_second: int | None = None,
		current_frame: int | None = None
	):
		"""
		환경 데이터를 초기화합니다.
		:param canvas: 캔버스
		:param frame_per_second: 초당 프레임 수
		:param current_frame: 현재 프레임
		:return:
		"""
		if canvas is not None:
			if not isinstance(canvas, np.ndarray):
				raise TypeError("canvas는 numpy.ndarray 타입이어야 합니다.")
			self.init_canvas = canvas
		self.dynamic_canvas = self.init_canvas.copy()
		self.dynamic_collision_mask = self.static_collision_mask.copy()
		if frame_per_second is not None:
			self.frame_per_second = frame_per_second
		if current_frame is not None:
			self.current_frame = current_frame


class ObjectManager:
	def __init__(self):
		self.__next_id: int = 1000  # ID 시작 번호
		self.__lock: threading.Lock = threading.Lock()  # 스레드 안전성
		self.__object_list: list[(int, int)] = []  # (priority, _id) 형태의 리스트
		self.__object_id_dict: dict[int, Object] = {}  # ID로 오브젝트를 빠르게 찾기 위한 딕셔너리

	def get_new_id(self, obj: Object) -> int:
		"""새로운 ID를 생성하고 반환합니다."""
		with self.__lock:  # 동시 접근 방지
			self.__next_id += 1

			# 튜플로 정렬 기준을 명시적으로 구성 (priority, _id)
			entry = (obj.priority, self.__next_id)
			bisect.insort(self.__object_list, entry)
			self.__object_id_dict[self.__next_id] = obj
			return self.__next_id

	def get_object_by_id(self, obj_id: int) -> Object | None:
		"""ID로 오브젝트를 검색합니다."""
		with self.__lock:
			return self.__object_id_dict.get(obj_id)

	def remove_object(self, obj: Object) -> None:
		"""오브젝트를 제거합니다."""
		with self.__lock:
			# ID로 오브젝트를 찾고, 리스트와 딕셔너리에서 제거
			if obj.id in self.__object_id_dict:
				del self.__object_id_dict[obj.id]
				entry = (obj.priority, obj.id)
				index = bisect.bisect_left(self.__object_list, entry)
				if index < len(self.__object_list) and self.__object_list[index] == entry:
					self.__object_list.pop(index)

	def step(self, env_data: EnvData) -> None:
		"""모든 오브젝트의 상태를 업데이트합니다."""
		with self.__lock:
			# priority 순으로 정렬된 오브젝트 리스트를 순회하며 step 호출
			for _, obj_id in self.__object_list:
				obj = self.__object_id_dict[obj_id]
				obj.step(env_data)

	def draw(self, env_data: EnvData) -> None:
		"""모든 오브젝트를 그립니다."""
		with self.__lock:
			# priority 순으로 정렬된 오브젝트 리스트를 순회하며 draw 호출
			for _, obj_id in self.__object_list:
				obj = self.__object_id_dict[obj_id]
				obj.draw(env_data)


class Object(ABC):
	pos: Point  # 오브젝트의 위치
	anchor: Point  # 오브젝트의 앵커 위치
	priority: int  # 낮을수록 먼저 계산되고, 먼저 그려짐
	image: np.ndarray  # 이미지
	collision_mask: np.ndarray  # 마스크
	_id: int

	def __init__(
		self,
		object_manager: ObjectManager,
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
		self._id = object_manager.get_new_id(self)

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
		:param env_data: 환경 데이터
		:return:
		"""
		pass

	def __str__(self):
		class_name = type(self).__name__
		return f"{class_name}(id={self.id}, pos={self.pos})"


