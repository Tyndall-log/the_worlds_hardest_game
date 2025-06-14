from pathlib import Path

from gymnasium import spaces
import numpy as np
import cv2
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from stage_map.map import MapData
from stage_map.env import Environment
# from model.AutoEncoder.SB_CNNPolicy import NatureCNNAutoEncoder
from model.AutoEncoder.resnet18 import ResNetAutoEncoder


class MapDataset(Dataset):
	def __init__(self, num_samples=1000):
		self.num_samples = num_samples
		self.map_data = MapData(None)
		self.output = np.zeros((1, 160, 256, 3), dtype=np.uint8)

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		map_dict = {
			"matrix": np.random.randint(0, 3, (12, 20)).tolist(),
		}
		self.map_data.draw_background(map_dict)
		raw_map = self.map_data.background_image  # shape: (H, W, 3)

		Environment.origin_image_random_crop_resize(
			raw_map[np.newaxis, ...],
			dst_batch_image=self.output,
			crop_offset=(Environment.crop_offset_x, Environment.crop_offset_y),
		)

		# RGBA 생성 (채널 4개)
		img = np.concatenate([self.output[0], np.full((160, 256, 1), 255, dtype=np.uint8)], axis=-1)
		img = img.transpose(2, 0, 1)  # (C, H, W)
		img = torch.tensor(img, dtype=torch.float32) / 255.0
		return img


@torch.no_grad()
def visualize_reconstruction(model, map_data, num_samples=3):
	model.eval()

	for i in range(num_samples):
		# 1. 랜덤 맵 생성
		map_dict = {
			"matrix": np.random.randint(0, 3, (12, 20)).tolist(),
		}
		map_data.draw_background(map_dict)
		raw_map = map_data.background_image  # (H, W, 3)

		# 2. 리사이즈 + RGBA 변환
		output = np.zeros((1, 160, 256, 3), dtype=np.uint8)
		Environment.origin_image_random_crop_resize(
			raw_map[np.newaxis, ...],
			dst_batch_image=output,
			crop_offset=(Environment.crop_offset_x, Environment.crop_offset_y),
		)
		img = np.concatenate([output[0], np.full((160, 256, 1), 255, dtype=np.uint8)], axis=-1)  # (160, 256, 4)
		img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0  # (1, 4, 160, 256)

		# 3. 복원
		img_tensor = img_tensor.to(next(model.parameters()).device)
		recon, _ = model(img_tensor)
		recon = recon.squeeze(0).cpu().numpy()  # (4, H, W)
		recon = np.clip(recon * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)  # (H, W, 4)

		# 4. 시각화
		original = img  # (H, W, 4)
		both = np.concatenate([original, recon], axis=1)  # 좌: 원본 / 우: 복원

		# 채널 순서 변경 (BGR로 변환)
		both = cv2.cvtColor(both, cv2.COLOR_RGBA2BGR)
		cv2.imshow(f"Reconstruction {i+1}", both)
		cv2.waitKey(0)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	observation_space = spaces.Box(low=0, high=255, shape=(4, 160, 256), dtype=np.uint8)
	# model = NatureCNNAutoEncoder(observation_space)
	model = ResNetAutoEncoder()
	save_file_name = "resnet18_autoencoder.pth"
	if Path(save_file_name).exists():
		model.load_state_dict(torch.load(save_file_name))
	device = torch.device(
		"cuda" if torch.cuda.is_available() else (
			"mps" if torch.backends.mps.is_available() else "cpu"
		)
	)
	model = model.to(device)

	dataset = MapDataset(num_samples=1000)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	# 🏋️ 학습 루프
	num_epochs = 10
	model.train()

	for epoch in range(num_epochs):
		total_loss = 0.0
		for batch in tqdm(dataloader, desc=f"[Epoch {epoch + 1}]"):
			batch = batch.to(device)
			recon, _ = model(batch)

			# recon = recon[..., 2:-2, 2:-2]
			# batch = batch[..., 2:-2, 2:-2]

			loss = criterion(recon, batch)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		avg_loss = total_loss / len(dataloader)
		print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

	# 모델 저장
	torch.save(model.state_dict(), save_file_name)

	visualize_reconstruction(model, dataset.map_data, num_samples=3)

