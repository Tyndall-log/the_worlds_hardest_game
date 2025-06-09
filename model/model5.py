import torch
import torch.nn as nn
from transformers import Mamba2Model, Mamba2Config

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x):
		identity = self.shortcut(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += identity
		return self.relu(out)


class ResNet18(nn.Module):
	def __init__(self, input_channels=3):
		super(ResNet18, self).__init__()
		# ResNet18의 각 단계 정의
		self.initial = nn.Sequential(
			nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)
		self.layer1 = nn.Sequential(
			ResidualBlock(32, 32),
			ResidualBlock(32, 32)
		)
		self.layer2 = nn.Sequential(
			ResidualBlock(32, 64, stride=2),
			ResidualBlock(64, 64)
		)
		self.layer3 = nn.Sequential(
			ResidualBlock(64, 128, stride=2),
			ResidualBlock(128, 128)
		)
		self.layer4 = nn.Sequential(
			ResidualBlock(128, 256, stride=2),
			ResidualBlock(256, 256)
		)

	def forward(self, x):
		# 각 단계에서의 출력 (특징 맵)을 저장
		features = []
		x = self.initial(x)      # 초기 레이어
		features.append(x)       # 첫 번째 특징 맵 저장
		x = self.layer1(x)       # 첫 번째 Residual Block
		features.append(x)       # 두 번째 특징 맵 저장
		x = self.layer2(x)       # 두 번째 Residual Block
		features.append(x)       # 세 번째 특징 맵 저장
		x = self.layer3(x)       # 세 번째 Residual Block
		features.append(x)       # 네 번째 특징 맵 저장
		x = self.layer4(x)       # 네 번째 Residual Block
		features.append(x)       # 마지막 특징 맵 저장

		return features  # 각 단계의 특징 맵을 반환

class ResNetMambaAgent(nn.Module):
	def __init__(self, image_channels=3, mamba_d_model=256, mamba_n_layers=6):
		super().__init__()
		self.image_encoder = ResNet18(input_channels=image_channels)

		# 최종 feature map을 받아서 펼치기 위한 처리
		self.final_conv = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),  # [B, C, 1, 1]
			nn.Flatten(),                 # [B, C]
			nn.Linear(256, mamba_d_model)  # 256 -> mamba_d_model
		)

		# Mamba 구성
		config = Mamba2Config(
			d_model=mamba_d_model,
			n_layers=mamba_n_layers,
			vocab_size=1,  # 텍스트 예측이 아니니까 아무 숫자
			use_cache=False
		)
		self.mamba = Mamba2Model(config)

		# PPO용 Actor/Critic 헤드
		self.actor = nn.Linear(mamba_d_model, 4)   # 4방향 이동 (상하좌우)
		self.critic = nn.Linear(mamba_d_model, 1)

	def forward(self, img_seq):
		"""
		img_seq: [B, T, C, H, W]   (Batch, Time, Channel, Height, Width)
		"""
		B, T, C, H, W = img_seq.shape
		img_seq = img_seq.view(B * T, C, H, W)    # 시간 축 병합

		# 1. 이미지 인코딩
		features = self.image_encoder(img_seq)
		x = features[-1]                          # 마지막 layer 사용
		x = self.final_conv(x)                    # [B*T, d_model]

		# 2. 시퀀스 준비
		x = x.view(B, T, -1)                      # 다시 [B, T, d_model]

		# 3. Mamba로 시퀀스 처리
		mamba_out = self.mamba(inputs_embeds=x).last_hidden_state  # [B, T, d_model]

		# 4. 마지막 step 출력만 사용
		last_hidden = mamba_out[:, -1, :]          # [B, d_model]

		# 5. Actor, Critic 출력
		action_logits = self.actor(last_hidden)   # [B, 4]
		value = self.critic(last_hidden)           # [B, 1]

		return action_logits, value.squeeze(-1)    # value: [B]


if __name__ == "__main__":
	# ResNet18 모델 생성 및 파라미터 계산
	model = ResNet18()
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total Parameters: {total_params}")

	# PPO 모델 생성 및 파라미터 계산
	model = PPOModel(3, 4)
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total Parameters: {total_params}")

	# PPO 모델 입력 테스트
	batch_size = 16
	input_channels = 3
	action_space = 9
	hidden_size = 256
	model = PPOModel(input_channels, action_space, hidden_size)

	# 입력 데이터 생성
	x = torch.randn(batch_size, input_channels, 1024 // 4, 640 // 4)  # (batch_size, channels, height, width)
	hidden_state = model.init_hidden_states(batch_size)

	# 모델 테스트
	# policy, value, hidden_state = model(x, hidden_state)
	# print(f"Policy: {policy.shape}, Value: {value.shape}, Hidden State: {hidden_state[0].shape}")

	# 모델 추론 속도 테스트
	import time
	device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
	model.to(device)
	model.eval()
	x = x.to(device)
	hidden_state = hidden_state.to(device)
	policy, value, hidden_state = model(x, hidden_state)  # 사전 초기화
	start_time = time.time()
	N = 10000
	for _ in range(N):
		print(_)
		policy, value, hidden_state = model(x, hidden_state)
	print(f"Inference Time: {(time.time() - start_time) * 1000 / N:.3f}ms")

