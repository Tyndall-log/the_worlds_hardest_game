from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import trange
from typing import List
from torch.utils.tensorboard import SummaryWriter

from stage_map.env import Environment
# from model.model7 import ResNetPolicy
from model.model6 import CNNPolicy

def compute_gae(rewards, values, gamma=0.997, lam=0.985):
	advantages = np.zeros_like(rewards)
	last_gae_lam = 0
	for t in reversed(range(len(rewards))):
		delta = rewards[t] + gamma * values[t + 1] - values[t]
		advantages[t] = last_gae_lam = delta + gamma * lam * last_gae_lam
	returns = advantages + values[:-1]
	return advantages, returns


@dataclass
class Transition:
	obs: np.ndarray
	action: int | np.int64
	log_prob: float | np.float32
	reward: float | np.float32
	value: float | np.float32
	done: bool | np.bool_


@dataclass
class Trajectory:
	env_num: int
	player_num: int
	observations: List[np.ndarray]
	actions: List[int]
	log_probs: List[float]
	rewards: List[float]
	values: List[float]
	dones: List[bool]
	advantages: np.ndarray
	returns: np.ndarray

	@classmethod
	def from_transitions(
		cls,
		env_num: int,
		player_num: int,
		transitions: List[Transition],
		gamma=0.997,
		lam=0.985
	):
		rewards = [t.reward for t in transitions]
		values = [t.value for t in transitions] + [0.0]
		advantages, returns = compute_gae(rewards, values, gamma, lam)
		return cls(
			env_num=env_num,
			player_num=player_num,
			observations=[t.obs for t in transitions],
			actions=[t.action for t in transitions],
			log_probs=[t.log_prob for t in transitions],
			rewards=rewards,
			values=values[:-1],
			dones=[t.done for t in transitions],
			advantages=advantages,
			returns=returns,
		)


class TrajectoryBuffer:
	def __init__(self, gamma=0.99, lam=0.95):
		self.buffer = []
		self.env_player_buffer = deque(maxlen=100)  # 최근 100개의 env-player 조합만 저장
		self.gamma = gamma
		self.lam = lam

	def add_trajectory(
		self,
		env_num: int,
		player_num: int,
		transitions: List[Transition],
	):
		traj = Trajectory.from_transitions(
			env_num=env_num,
			player_num=player_num,
			transitions=transitions,
			gamma=self.gamma,
			lam=self.lam,
		)
		self.buffer.append(traj)

	def total_steps(self):
		return sum(len(t.rewards) for t in self.buffer)

	def get_rollout(self, max_steps):
		rollout = []
		used_trajectories = []
		steps = 0
		for t in self.buffer:
			rollout.append(t)
			used_trajectories.append(t)
			steps += len(t.rewards)
			if steps >= max_steps:
				break
		for t in used_trajectories:
			self.buffer.remove(t)
		return rollout

	def is_ready(self, threshold):
		return self.total_steps() >= threshold


import torch
from torch.distributions import Categorical

class PreviousAction:
	def __init__(self, k=3):
		self.k = k
		self.current_action: int | None = None
		self.steps_remaining: int = 0

	def reset(self, action: int):
		self.current_action = action
		self.steps_remaining = self.k

	def step(self, logits: torch.Tensor) -> tuple[int, float]:
		probs_cpu = torch.softmax(logits, dim=-1)
		dist = Categorical(probs_cpu)

		need_resample = (
			self.current_action is None or
			self.steps_remaining <= 0 or
			probs_cpu[self.current_action].item() < probs_cpu.max().item() - 0.2
		)

		if need_resample:
			action = dist.sample().item()
			self.reset(action)
		else:
			action = self.current_action
			self.steps_remaining -= 1

		logprob = dist.log_prob(torch.tensor(action)).item()
		return action, logprob


@torch.no_grad()
def select_action(policy, obs, previous_action_list: list[PreviousAction]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	logits, value = policy(obs)
	logits_cpu = logits.cpu()  # logits 전체를 한번에 CPU로 이동

	actions = []
	logprobs = []

	for i, pa in enumerate(previous_action_list):
		action, logprob = pa.step(logits_cpu[i])  # 이미 CPU에 있음
		actions.append(action)
		logprobs.append(logprob)

	return (
		np.array(actions, dtype=np.int64),
		np.array(logprobs, dtype=np.float32),
		value.cpu().numpy()
	)


def train(checkpoint_path=None):
	root_path = Path(__file__).parent
	model_train_time = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	log_folder = root_path / "logs" / model_train_time
	writer = SummaryWriter(log_dir=log_folder.as_posix())

	save_path = root_path / "checkpoints" / model_train_time
	save_path.mkdir(exist_ok=True)

	save_interval = 10000  # 1만 스텝마다 저장
	global_step = 0  # 맨 위에 선언

	n_envs = 4
	n_players = 4  # 플레이어 수
	map_list = ["map1.json", "map2.json", "map3.json", "map4.json"]
	envs = [
		Environment(
			name=f"env_{i}_{map_list[i].split('.')[0]}",
			map_path=root_path / "stage_map/stage/stage0" / map_list[i],
			player_num=n_players,
		) for i in range(n_envs)
	]
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

	model = CNNPolicy().to(device)
	stack_frame = 1
	if checkpoint_path is not None and checkpoint_path.exists():
		print(f"Loading model from {checkpoint_path}")
		model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
	buffer = TrajectoryBuffer(gamma=0.997, lam=0.985)
	obs_batch = [env.reset()[0] for env in envs]
	obs_tensor = torch.tensor(np.concatenate(obs_batch), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
	obs_tensor = obs_tensor.repeat(1, stack_frame, 1, 1)  # (N, C, H, W) 형태로 변환

	max_steps_per_update = 2048 // 4

	episode_transitions = [[[] for _ in range(n_players)] for _ in range(n_envs)]
	previous_actions = [PreviousAction(k=6) for _ in range(n_envs * n_players)]

	for step in trange(int(1e6)):
		obs_tensor_temp = torch.tensor(np.concatenate(obs_batch), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
		obs_tensor[:, :-4, ...] = obs_tensor[:, 4:, ...].clone()  # 프레임 스택 업데이트
		obs_tensor[:, -4:, ...] = obs_tensor_temp
		actions, logprobs, values = select_action(model, obs_tensor, previous_action_list=previous_actions)

		next_obs_batch, rewards, terminateds, truncateds, infos = zip(
			*(envs[i].step(actions[n_players*i:n_players*(i+1)]) for i in range(n_envs))
		)

		for e in range(n_envs):
			for p in range(n_players):
				transition = Transition(
					obs=obs_batch[e][p].copy(),
					action=actions[e * n_players + p],
					log_prob=logprobs[e * n_players + p],
					reward=rewards[e][p],
					value=values[e * n_players + p],
					done=terminateds[e][p] or truncateds[e][p],
				)
				# print(obs_batch[e][p].mean())
				episode_transitions[e][p].append(transition)

				if transition.done:
					buffer.add_trajectory(
						env_num=e,
						player_num=p,
						transitions=episode_transitions[e][p],
					)
					episode_transitions[e][p] = []
					obs_batch[e][p] = envs[e].reset_player(p)[0]
					temp = torch.tensor(obs_batch[e][p], dtype=torch.float32).permute(2, 0, 1).to(device)
					obs_tensor[e * n_players + p, :, :, :] = temp.repeat(stack_frame, 1, 1)
				else:
					obs_batch[e][p] = next_obs_batch[e][p]

		if buffer.is_ready(max_steps_per_update):
			rollout = buffer.get_rollout(max_steps_per_update)
			global_step += sum(len(t.rewards) for t in rollout)  # 수집 기준으로만 증가!
			train_step(model, optimizer, rollout, device, writer=writer, global_step=global_step)

		if step % save_interval == save_interval - 1:
			torch.save(model.state_dict(), save_path / f"model_step_{global_step}.pt")
			print(f"✅ Model saved at step {global_step}")


def train_step(model, optimizer, rollout, device, writer, global_step, epochs=4, minibatch_size=64):
	obs = torch.tensor(np.concatenate([t.observations for t in rollout]), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
	actions = torch.tensor(np.concatenate([t.actions for t in rollout]), dtype=torch.int64).to(device)
	logprobs = torch.tensor(np.concatenate([t.log_probs for t in rollout]), dtype=torch.float32).to(device)
	advantages = torch.tensor(np.concatenate([t.advantages for t in rollout]), dtype=torch.float32).to(device)
	returns = torch.tensor(np.concatenate([t.returns for t in rollout]), dtype=torch.float32).to(device)

	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

	for _ in range(epochs):
		inds = np.arange(len(obs))
		np.random.shuffle(inds)
		for start in range(0, len(obs), minibatch_size):
			end = start + minibatch_size
			mb_inds = inds[start:end]

			logits, values = model(obs[mb_inds])
			dist = Categorical(logits=logits)
			new_logprobs = dist.log_prob(actions[mb_inds])
			ratio = (new_logprobs - logprobs[mb_inds]).exp()

			pg_loss = -torch.min(
				ratio * advantages[mb_inds],
				torch.clamp(ratio, 0.8, 1.2) * advantages[mb_inds],
			).mean()
			v_loss = ((returns[mb_inds] - values) ** 2).mean()
			entropy = dist.entropy().mean()

			loss = pg_loss + 0.5 * v_loss - 0.01 * entropy
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			writer.add_scalar("loss/total", loss.item(), global_step)
			writer.add_scalar("loss/policy", pg_loss.item(), global_step)
			writer.add_scalar("loss/value", v_loss.item(), global_step)
			writer.add_scalar("loss/entropy", entropy.item(), global_step)


if __name__ == "__main__":
	# checkpoint_path = Path(__file__).parent / "checkpoints" / "model_step_311950.pt"
	checkpoint_path = None  # None이면 새로 학습
	train(checkpoint_path)
