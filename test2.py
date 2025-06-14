
import torch
from torchvision.models import resnet18
from torch import nn

model = resnet18()
features = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
	nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
	nn.ReLU(),
	nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
	nn.ReLU(),
	nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
	nn.ReLU(),
	nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
	nn.ReLU(),
	nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, output_padding=1),
)

x = torch.randn(1, 3, 160, 256)
for i, layer in enumerate(features):
    x = layer(x)
    print(f"After layer {i}: {x.shape}")


import torch
import torch.nn as nn

x = torch.ones(1, 1, 3, 3)

convT = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
with torch.no_grad():
    convT.weight[:] = 1.0  # 전부 1로 초기화
    y = convT(x)
    print(y[0, 0])  # 출력 확인