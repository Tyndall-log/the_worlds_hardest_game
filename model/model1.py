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
			nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)
		self.layer1 = nn.Sequential(
			ResidualBlock(16, 16),
			ResidualBlock(16, 16)
		)
		self.layer2 = nn.Sequential(
			ResidualBlock(16, 16, stride=2),
			ResidualBlock(16, 16)
		)
		self.layer3 = nn.Sequential(
			ResidualBlock(16, 16, stride=2),
			ResidualBlock(16, 16)
		)
		self.layer4 = nn.Sequential(
			ResidualBlock(16, 16, stride=2),
			ResidualBlock(16, 16)
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


# PPO 모델 (ResNet18 + LSTM 기반)
class PPOModel(nn.Module):
	def __init__(self, input_channels, action_space, hidden_size=256):
		super(PPOModel, self).__init__()
		self.cnn = ResNet18(input_channels)
		self.fc = nn.Linear(16 * 36 * 20, 512)
		self.lstm = nn.LSTM(512, hidden_size, batch_first=True)

		# Actor-Critic Heads
		self.actor = nn.Linear(hidden_size, action_space)  # 행동 확률 분포
		self.critic = nn.Linear(hidden_size, 1)  # 상태 가치

	def forward(self, x, hidden_state):
		"""
		x: 입력 데이터 (batch_size, channels, height, width)
		hidden_state: LSTM의 초기 히든 상태와 셀 상태
		"""
		# CNN 처리
		features = self.cnn(x)  # CNN 출력 (batch_size, feature_size)
		final_features = features[-1]  # 마지막 레이어의 출력 (batch_size, 512, 36, 20)
		flattened_features = final_features.view(final_features.size()[0], -1)  # (batch_size, channels * height * width)
		fc_out = self.fc(flattened_features)  # FC 출력 (batch_size, 512)

		# LSTM 처리
		lstm_input = fc_out.unsqueeze(1)
		lstm_out, hidden_state = self.lstm(lstm_input, hidden_state)

		# Actor-Critic Heads
		policy = self.actor(lstm_out[:, -1, :])  # 행동 확률 분포
		value = self.critic(lstm_out[:, -1, :])  # 상태 가치
		return policy, value, hidden_state

	def init_hidden_state(self, batch_size):
		"""
		LSTM 초기 히든 상태와 셀 상태 생성
		"""
		hidden_state = (
			torch.zeros(1, batch_size, self.lstm.hidden_size),  # LSTM 히든 상태 초기화
			torch.zeros(1, batch_size, self.lstm.hidden_size)   # LSTM 셀 상태 초기화
		)
		return hidden_state


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
	# sequence_length = 5
	input_channels = 3
	action_space = 9
	hidden_size = 256
	model = PPOModel(input_channels, action_space, hidden_size)

	# 입력 데이터 생성
	x = torch.randn(batch_size, input_channels, 1152, 640)  # (batch_size, channels, height, width)
	hidden_state = model.init_hidden_state(batch_size)

	# 모델 테스트
	policy, value, hidden_state = model(x, hidden_state)
	print(f"Policy: {policy.shape}, Value: {value.shape}, Hidden State: {hidden_state[0].shape}")

