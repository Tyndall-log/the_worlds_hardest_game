import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# ==== 1. OpenCV 버전 (LINE_AA 원 그리기) ====
cv_canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255
cv2.circle(
	cv_canvas,
	center=(150, 150),
	radius=80,
	color=(0, 0, 0),
	thickness=-1,
	lineType=cv2.LINE_AA
)

# ==== 2. PyTorch 버전 ====
def draw_filled_circle_aa(image: torch.Tensor, center: tuple, radius: float, color: tuple, aa_width: float = 1.0):
	"""
	image: (H, W, 3), float32 in [0, 1], CUDA tensor
	center: (x, y)
	radius: float
	color: (R, G, B)
	aa_width: 경계 부드러움 정도 (보통 1.0 ~ 2.0)
	"""
	H, W, _ = image.shape
	y, x = torch.meshgrid(
		torch.arange(H, dtype=torch.float32, device=image.device),
		torch.arange(W, dtype=torch.float32, device=image.device),
		indexing="ij"
	)
	dist = torch.sqrt((x - center[0])**2 + (y - center[1])**2)

	# anti-aliased alpha mask
	alpha = torch.clamp(1.0 - (dist - radius) / aa_width, 0.0, 1.0)
	alpha[dist <= radius - aa_width] = 1.0  # 내부는 완전 채움

	for c in range(3):
		image[..., c] = image[..., c] * (1 - alpha) + color[c] * alpha

	return image

# torch 원 이미지 생성
torch_canvas = torch.ones((300, 300, 3), dtype=torch.float32, device="mps")
torch_result = draw_filled_circle_aa(torch_canvas, center=(150.0, 150.0), radius=80.0, color=(0.0, 0.0, 0.0), aa_width=1.5)
torch_result_np = (torch_result.cpu().numpy() * 255).astype(np.uint8)

# ==== 3. 시각화 ====
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("OpenCV (LINE_AA)")
plt.imshow(cv2.cvtColor(cv_canvas, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("PyTorch (Anti-Aliased)")
plt.imshow(torch_result_np)
plt.axis('off')

plt.tight_layout()
plt.show()

import time

# 테스트 반복 수
N = 100

# 1. OpenCV 성능 측정
canvas = np.ones((300, 300, 3), dtype=np.uint8) * 255
start_time = time.perf_counter()
for _ in range(N):
	cv2.circle(canvas, center=(150, 150), radius=80, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
cv_time = (time.perf_counter() - start_time) * 1000 / N  # ms
print(f"OpenCV avg: {cv_time:.3f} ms per call")

# 2. PyTorch 성능 측정 (GPU 연산)
torch.cuda.synchronize() if torch.cuda.is_available() else None
canvas = torch.ones((300, 300, 3), dtype=torch.float32, device="mps")  # 또는 "cuda"
start_time = time.perf_counter()
for _ in range(N):
	draw_filled_circle_aa(canvas, center=(150.0, 150.0), radius=80.0, color=(0.0, 0.0, 0.0), aa_width=1.5)
torch.cuda.synchronize() if torch.cuda.is_available() else None
torch_time = (time.perf_counter() - start_time) * 1000 / N  # ms
print(f"PyTorch avg: {torch_time:.3f} ms per call")