import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym  # Gymnasium을 사용
import numpy as np

# PPO 하이퍼파라미터
GAMMA = 0.99  # 감가율
LAMBDA = 0.95  # Generalized Advantage Estimation (GAE)의 람다 값
EPSILON = 0.2  # PPO 클리핑 범위
LEARNING_RATE = 3e-4  # 학습률
ENTROPY_COEF = 0.01  # 탐험을 위한 엔트로피 보상 계수
CRITIC_COEF = 0.5  # Critic 손실 계수
BATCH_SIZE = 64  # 미니배치 크기
EPOCHS = 4  # PPO 최적화를 반복할 횟수

# PPO 모델 (ResNet + LSTM 기반)
class PPOModel(nn.Module):
    def __init__(self, input_channels, action_space):
        super(PPOModel, self).__init__()
        # ResNet 스타일 CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),  # 입력 크기 3x224x224 가정
            nn.ReLU(),
        )
        # 시퀀스 정보를 처리하는 LSTM
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        # Actor-Critic 출력 레이어
        self.actor = nn.Linear(256, action_space)  # 행동 확률 분포
        self.critic = nn.Linear(256, 1)  # 상태 가치

    def forward(self, x, hidden_state):
        # 입력 크기: (배치, 시퀀스 길이, 채널, 높이, 너비)
        batch_size, sequence_length, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # CNN 처리를 위해 배치와 시퀀스를 병합
        features = self.cnn(x)
        features = features.view(batch_size, sequence_length, -1)  # 배치와 시퀀스를 복원
        lstm_out, hidden_state = self.lstm(features, hidden_state)  # LSTM 처리
        policy = self.actor(lstm_out)  # 정책 출력
        value = self.critic(lstm_out)  # 상태 가치 출력
        return policy, value, hidden_state

    def init_hidden_state(self, batch_size):
        # LSTM 초기 은닉 상태와 셀 상태 생성
        return (torch.zeros(1, batch_size, 256), torch.zeros(1, batch_size, 256))

# PPO 알고리즘 클래스
class PPO:
    def __init__(self, model, action_space):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.action_space = action_space

    def select_action(self, state, hidden_state):
        # 행동 선택 및 상태 가치 계산
        with torch.no_grad():
            logits, value, hidden_state = self.model(state, hidden_state)
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)  # 마지막 시퀀스에서 행동 확률 계산
            action = torch.multinomial(probabilities, 1).item()  # 확률적으로 행동 선택
        return action, hidden_state, value[:, -1, :].squeeze()

    def compute_loss(self, states, actions, advantages, returns, log_probs_old, hidden_states):
        # PPO 손실 함수 계산
        logits, values, _ = self.model(states, hidden_states)
        values = values[:, -1, :].squeeze()  # 상태 가치
        logits = logits[:, -1, :]  # 정책 로그확률 계산
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # PPO 손실 계산
        ratio = torch.exp(log_probs - log_probs_old)  # 정책 비율
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Critic 손실 계산
        critic_loss = (returns - values).pow(2).mean()

        # 엔트로피 손실 (탐험 유도)
        entropy = -torch.sum(torch.softmax(logits, dim=-1) * log_probs, dim=-1).mean()

        # 총 손실 계산
        loss = actor_loss + CRITIC_COEF * critic_loss - ENTROPY_COEF * entropy
        return loss

    def train(self, trajectories):
        # PPO 알고리즘 학습
        states, actions, rewards, dones, log_probs, values, hidden_states = zip(*trajectories)

        # Returns와 Advantages 계산
        returns = self.compute_returns(rewards, dones, values)
        advantages = returns - torch.cat(values)

        # 미니배치 기반으로 최적화 반복
        for _ in range(EPOCHS):
            loss = self.compute_loss(
                torch.cat(states),
                torch.cat(actions),
                advantages,
                returns,
                torch.cat(log_probs),
                torch.cat(hidden_states)
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards, dones, values):
        # GAE를 사용하지 않는 단순 리턴 계산
        returns = []
        G = 0
        for step in reversed(range(len(rewards))):
            G = rewards[step] + GAMMA * G * (1 - dones[step])
            returns.insert(0, G)
        return torch.tensor(returns).float()

# 학습 루프
def train(env, model, ppo, num_episodes=1000):
    for episode in range(num_episodes):
        state, _ = env.set()
        hidden_state = model.init_hidden_state(1)
        done = False
        episode_reward = 0
        trajectories = []

        while not done:
            state_tensor = torch.tensor(state).unsqueeze(0).unsqueeze(0).float()
            action, hidden_state, value = ppo.select_action(state_tensor, hidden_state)
            next_state, reward, done, _, _ = env.step(action)

            log_prob = torch.log(torch.softmax(model(state_tensor, hidden_state)[0][:, -1, :], dim=-1)[0, action])
            trajectories.append((state_tensor, torch.tensor([action]), reward, done, log_prob, value, hidden_state))

            state = next_state
            episode_reward += reward

        ppo.train(trajectories)
        print(f"Episode {episode}, Reward: {episode_reward}")

# 환경, 모델, PPO 초기화
env = gym.make("CartPole-v1")  # 미로 환경으로 교체 가능
action_space = env.action_space.n
input_channels = 3  # RGB 이미지

model = PPOModel(input_channels, action_space)
ppo = PPO(model, action_space)

train(env, model, ppo, num_episodes=500)