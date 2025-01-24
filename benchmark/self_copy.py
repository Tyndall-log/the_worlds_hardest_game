import torch
import kornia
import time

# MPS 디바이스 설정
if torch.backends.mps.is_available():
	device = torch.device("mps")
else:
	raise RuntimeError("MPS device is not available.")

# GPU 텐서 생성
tensor = torch.rand(3, 1000, 1000, device=device)  # Channels x Height x Width

# GPU 내에서 복사 속도 측정
num_iterations = 1000
start_time = time.time()
copied_tensor = tensor.clone()  # GPU 내에서 복사
for _ in range(num_iterations):
	copied_tensor = copied_tensor.clone()  # GPU 내에서 복사
	copied_tensor[0, 0, 0] += 0.1
elapsed_time = (time.time() - start_time) * 1000 / num_iterations
print(copied_tensor[0, 0, 0])

print(f"Average copy elapsed time: {elapsed_time:.3f} ms")