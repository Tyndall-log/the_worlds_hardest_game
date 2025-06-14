from stable_baselines3.common.torch_layers import NatureCNN
from gymnasium.spaces import Box
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class NatureCNNDecoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 64, 3, stride=1, padding=0),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 4, 8, stride=4, padding=0),
		)

	def forward(self, x):
		x = self.deconv(x)
		x = torch.tanh(x / 2) * 2 + 1
		return x


class NatureCNNAutoEncoder(nn.Module):
	def __init__(self, observation_space):
		super().__init__()
		nc = NatureCNN(observation_space)
		self.encoder = nn.Sequential(*list(nc.cnn.children())[:-1])  # Exclude the last layer (flatten)
		self.decoder = NatureCNNDecoder()

	def forward(self, obs):
		z = self.encoder(obs)
		recon = self.decoder(z)
		input_h, input_w = obs.shape[-2:]
		recon_h, recon_w = recon.shape[-2:]
		x_pad = (input_w - recon_w) // 2
		y_pad = (input_h - recon_h) // 2
		recon = F.pad(recon, (x_pad, x_pad, y_pad, y_pad), mode='replicate')
		return recon, z

if __name__ == "__main__":
	observation_space = Box(low=0, high=255, shape=(4, 160, 256), dtype=np.uint8)
	model = NatureCNNAutoEncoder(observation_space)

	# Test the model with a random observation
	random_obs = torch.randint(0, 256, (1, 4, 160, 256), dtype=torch.uint8)
	recon, z = model(random_obs.float() / 255.0)  # Normalize to [0, 1]
	print("Reconstructed shape:", recon.shape)
	print("Latent representation shape:", z.shape)