import numpy as np


class MaskInfo:
	class MaskLayer:
		WALL = 0b0000_0001
		# PLAYER = 0b0000_0010
		CHECKPOINT_ZONE = 0b0000_0100
		LETTERBOX = 0b0000_1000
		BALL = 0b0001_0000

	def __init__(self, mask_image: np.ndarray):
		self.mask_image = mask_image

	def __str__(self):
		return f"MaskInfo({self.mask_image.shape})"