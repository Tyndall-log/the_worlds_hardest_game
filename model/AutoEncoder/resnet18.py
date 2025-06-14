import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models


class ResNetAutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = torchvision.models.resnet18(weights=None)
		self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # remove the last two layers (avgpool and fc)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 4, 7, stride=2, padding=3, output_padding=1),
		)

	def forward(self, x):
		em = self.encoder(x)
		x = self.decoder(em)
		recon = torch.tanh(x / 2) + 0.5
		return recon, em

if __name__ == "__main__":
	model = ResNetAutoEncoder()

	# Test the model with a random observation
	random_obs = torch.randint(0, 256, (1, 4, 160, 256), dtype=torch.uint8)
	recon, z = model(random_obs.float() / 255.0)  # Normalize to [0, 1]
	print("Reconstructed shape:", recon.shape)
	print("Latent representation shape:", z.shape)