import torch
import torch.nn as nn
import torchvision.models as models

class ResNetPolicy(nn.Module):
	def __init__(self, in_channels=12, num_actions=9):
		super().__init__()

		# torchvision ResNet18 불러오기
		resnet = models.resnet18(weights=None)

		# 입력 채널 수정 (기본값은 3)
		if in_channels != 3:
			resnet.conv1 = nn.Conv2d(
				in_channels, 64,
				kernel_size=7,
				stride=2,
				padding=3,
				bias=False,
			)

		# FC 제거하고 feature extractor로 사용
		self.backbone = nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
			resnet.maxpool,
			resnet.layer1,
			resnet.layer2,
			resnet.layer3,
			resnet.layer4,
			nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
			nn.Flatten(),  # [B, 512]
		)

		self.actor = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, num_actions)
		)
		self.critic = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, x):
		x = x / 255.0
		feat = self.backbone(x)
		return self.actor(feat), self.critic(feat).squeeze(-1)