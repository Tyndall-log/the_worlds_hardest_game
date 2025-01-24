import torch
import torch.nn as nn


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


# PPO 모델
class PPOModel(nn.Module):
	def __init__(self, input_channels, action_space, spatial_hidden_size=256, temporal_hidden_size=256):
		super(PPOModel, self).__init__()
		# ResNet18으로 공간 특징 추출
		self.resnet_cnn = ResNet18(input_channels)

		# 첫 번째 LSTM: 공간 정보를 시퀀스로 해석
		self.spatial_lstm = nn.LSTM(256, spatial_hidden_size, batch_first=True)
		self.static_spatial_hidden = self._init_static_spatial_hidden(1)

		# 두 번째 LSTM: 장기 정보를 보관
		self.temporal_lstm = nn.LSTM(spatial_hidden_size, temporal_hidden_size, batch_first=True)

		# Actor-Critic Heads
		self.actor_fc = nn.Sequential(
			nn.Linear(temporal_hidden_size, 128),
			nn.ReLU(),
			nn.Linear(128, action_space)
		)
		self.critic_fc = nn.Sequential(
			nn.Linear(temporal_hidden_size, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	def forward(self, x, temporal_hidden):
		"""
		x: 입력 데이터 (batch_size, channels, height, width)
		spatial_hidden: 첫 번째 LSTM의 초기 히든 상태
		temporal_hidden: 두 번째 LSTM의 초기 히든 상태
		"""
		# ResNet18으로 공간 특징 추출
		features = self.resnet_cnn(x)  # (batch_size, 16, H, W)

		# 공간 정보를 시퀀스로 변환
		batch_size, channels, height, width = features[-1].size()
		spatial_sequence = features[-1].permute(0, 2, 3, 1).view(batch_size, height * width, channels)  # (batch_size, H*W, channels)

		# 첫 번째 LSTM 처리
		if self.static_spatial_hidden[0].shape[1] != batch_size:
			self.static_spatial_hidden = self._init_static_spatial_hidden(batch_size)
		if self.static_spatial_hidden[0].device != x.device:
			self.static_spatial_hidden = (
				self.static_spatial_hidden[0].to(x.device),
				self.static_spatial_hidden[1].to(x.device)
			)
		sshh = self.static_spatial_hidden[0].detach()
		sshc = self.static_spatial_hidden[1].detach()
		# ss = spatial_sequence.cpu().detach().numpy()
		# ss = torch.tensor(ss, dtype=torch.float32).to(x.device)
		with torch.no_grad():
			ss = spatial_sequence.detach().clone()
			# ss = torch.zeros(batch_size, height * width, 256).to(x.device)
			spatial_out, _ = self.spatial_lstm(ss)  # (batch_size, H*W, spatial_hidden_size)

		# 두 번째 LSTM 처리
		# temporal_out, _ = self.temporal_lstm(spatial_out[:, -1:, :], temporal_hidden)  # (batch_size, H*W, temporal_hidden_size)
		# _, temporal_hidden = self.temporal_lstm(spatial_out[:, -1:, :].detach(), temporal_hidden)  # Detach ResNet
		# temporal_out, temporal_hidden = self.temporal_lstm(spatial_out[:, -1:, :], temporal_hidden)  # (batch_size, H*W, temporal_hidden_size)
		# temporal_out, temporal_hidden = self.temporal_lstm(spatial_out[:, -1:, :].detach(), temporal_hidden)  # Detach ResNet

		# Actor-Critic Heads
		# temporal_out = spatial_out.squeeze(1)  # (batch_size, temporal_hidden_size)
		# policy = self.actor_fc(temporal_out)  # 행동 확률 분포 (batch_size, action_space)
		# value = self.critic_fc(temporal_out)  # 상태 가치 (batch_size, 1)

		return {}, {}, temporal_hidden
		# return policy, value, temporal_hidden

	def _init_static_spatial_hidden(self, batch_size):
		"""
		고정된 첫 번째 LSTM의 초기 히든 상태와 셀 상태 생성
		"""
		spatial_hidden = (
			torch.zeros(1, batch_size, self.spatial_lstm.hidden_size),  # 고정된 히든 상태
			torch.zeros(1, batch_size, self.spatial_lstm.hidden_size)  # 고정된 셀 상태
		)
		return spatial_hidden

	def init_hidden_states(self, batch_size):
		"""
		LSTM 초기 히든 상태와 셀 상태 생성
		"""
		# 두 번째 LSTM 히든 상태
		temporal_hidden = (
			torch.zeros(1, batch_size, self.temporal_lstm.hidden_size),
			torch.zeros(1, batch_size, self.temporal_lstm.hidden_size)
		)
		return temporal_hidden


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
	policy, value, hidden_state = model(x, hidden_state)
	# print(f"Policy: {policy.shape}, Value: {value.shape}, Hidden State: {hidden_state[0].shape}")

	# 모델 추론 속도 테스트
	import time
	model.eval()
	# model.train()
	device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
	model = model.to(device)
	x = x.to(device)
	hidden_state = [t.to(device) for t in hidden_state]
	policy, value, hidden_state = model(x, hidden_state)  # 사전 초기화
	start_time = time.time()
	N = 10000
	for _ in range(N):
		print(_)
		policy, value, hidden_state = model(x, hidden_state)
	print(f"Inference Time: {(time.time() - start_time) * 1000 / N:.3f}ms")

