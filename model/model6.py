import torch
from torch import nn

class CNNPolicy(nn.Module):
	def __init__(self, in_channels=4, num_actions=9):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Flatten(),
		)
		with torch.no_grad():
			out_dim = self.features(torch.zeros(1, in_channels, 160, 256)).shape[1]

		self.actor = nn.Sequential(
			nn.Linear(out_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
		)
		self.critic = nn.Sequential(
			nn.Linear(out_dim, 512), nn.ReLU(), nn.Linear(512, 1)
		)

	def forward(self, x):
		x = x / 255.0
		feat = self.features(x)
		return self.actor(feat), self.critic(feat).squeeze(-1)