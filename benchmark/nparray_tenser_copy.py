import numpy as np
import torch
import time

# MPS 디바이스 설정
if torch.backends.mps.is_available():
	device = torch.device("mps")
else:
	raise RuntimeError("MPS device is not available.")

# 데이터 초기화
np_array = np.random.rand(1000, 1000).astype(np.float32)  # 1000x1000 float32 배열

# numpy -> torch.Tensor 변환 속도 측정
start_time = time.time()
torch_tensor = torch.from_numpy(np_array).to(device)
torch_to_device_time = time.time() - start_time

# torch.Tensor -> numpy 변환 속도 측정
start_time = time.time()
np_array_back = torch_tensor.cpu().numpy()
tensor_to_numpy_time = time.time() - start_time

print(f"Numpy to Torch Tensor (MPS): {torch_to_device_time*1000:.3f}ms")
print(f"Torch Tensor (MPS) to Numpy: {tensor_to_numpy_time*1000:.3f}ms")
