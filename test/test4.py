import torch
import torch.nn as nn
import time


def test_maxpool2d_no_leak():
	N, C, H, W = 64, 32, 256, 256
	iters = 5

	# 모델 정의
	model = torch.nn.Sequential(
		nn.Conv2d(32, 32, 3),
		nn.MaxPool2d((2, 2)),
	).to('mps')

	# 입력 데이터 생성
	input = torch.rand(N, C, H, W, device='mps')

	# 워밍업
	model(input)
	torch.mps.empty_cache()
	time.sleep(6)

	# 테스트 시작
	driver_before = torch.mps.driver_allocated_memory()
	print(f"Memory before test: {driver_before} bytes")

	for _ in range(iters):
		output = model(input)
		loss = output.sum()
		loss.backward()
	torch.mps.empty_cache()
	time.sleep(6)

	driver_after = torch.mps.driver_allocated_memory()
	print(f"Memory after test: {driver_after} bytes")
	print(f"Memory difference: {driver_after - driver_before} bytes")


def test_linear_no_leak():
	N, C, H, W = 64, 16, 64, 64
	iters = 10

	# 모델 정의
	model = nn.Linear(H, W, device='mps')

	# 입력 데이터 생성
	input = torch.rand(N, C, H, W, device='mps')

	# 워밍업
	model(input)
	torch.mps.empty_cache()
	time.sleep(6)

	# 테스트 시작
	driver_before = torch.mps.driver_allocated_memory()
	print(f"Memory before test: {driver_before} bytes")

	for _ in range(iters):
		model(input)
	torch.mps.empty_cache()

	time.sleep(6)
	driver_after = torch.mps.driver_allocated_memory()
	print(f"Memory after test: {driver_after} bytes")
	predicted_leaked = iters * 4 * N * C * H * W
	print(f"Predicted memory leak: {predicted_leaked} bytes")
	print(f"Memory difference: {driver_after - driver_before} bytes")


if __name__ == "__main__":
	print("Running test_maxpool2d_no_leak...")
	test_maxpool2d_no_leak()

	print("\nRunning test_linear_no_leak...")
	test_linear_no_leak()