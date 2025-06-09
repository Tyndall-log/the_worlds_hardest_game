from pathlib import Path
from datetime import datetime
import time
from dataclasses import dataclass
from collections import deque
from threading import Lock, Thread
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import trange
from typing import List
from torch.utils.tensorboard import SummaryWriter

from stage_map.env import Environment
from model.model7 import ResNetPolicy
from model.model6 import CNNPolicy

Policy = ResNetPolicy

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
	advantage: float | np.float32 = 0.0
	returns: float | np.float32 = 0.0


class TrajectoryBuffer:
	def __init__(self, env_num, player_num, stack_frame, gamma=0.99, lam=0.95):
		# self.buffer = []
		self.env_num = env_num
		self.player_num = player_num
		self.stack_frame = stack_frame
		self.env_player_buffer: list[list[Transition]] = [[] for _ in range(env_num * player_num)]
		self.env_player_buffer_ready_next_index = [0 for _ in range(env_num * player_num)]
		self.locks = [Lock() for _ in range(env_num * player_num)]
		self._all_ready_count = 0
		self.gamma = gamma
		self.lam = lam
	
	def add_transition(
		self,
		env_idx: int,
		player_idx: int,
		transition: Transition,
	):
		index = env_idx * self.player_num + player_idx
		with self.locks[index]:
			self.env_player_buffer[index].append(transition)

			if transition.done:
				# advantages, returns 계산
				ready_last = self.env_player_buffer_ready_next_index[index]
				transitions = self.env_player_buffer[index][ready_last:]
				rewards = [t.reward for t in transitions]
				values = [t.value for t in transitions] + [0.0]
				obses = [t.obs for t in transitions]
				advantages, returns = compute_gae(rewards, values, self.gamma, self.lam)
				for i, t in enumerate(transitions):
					obs = []
					for f in range(1-self.stack_frame, 1):
						obs.append(obses[max(0, i + f)])
					t.obs = np.concatenate(obs, axis=-1)  # stack frame
					t.advantage = advantages[i]
					t.returns = returns[i]
				self.env_player_buffer_ready_next_index[index] = len(self.env_player_buffer[index])
				self._all_ready_count = min(self.env_player_buffer_ready_next_index)

	@property
	def all_ready_count(self):
		return self._all_ready_count

	def get_rollout(self, max_steps):
		for lock in self.locks:
			lock.acquire()
		try:
			if self.all_ready_count < max_steps:
				raise ValueError(f"Buffer is not ready. Minimum ready index: {self.all_ready_count}, required: {max_steps}")
			rollout = []  # count = env_num * player_num * max_steps
			for step in range(max_steps):
				for i, buffer in enumerate(self.env_player_buffer):
					rollout.append(buffer[step])
			for i in range(len(self.env_player_buffer)):
				self.env_player_buffer[i] = self.env_player_buffer[i][max_steps:]
				self.env_player_buffer_ready_next_index[i] -= max_steps
			self._all_ready_count = min(self.env_player_buffer_ready_next_index)
			return rollout
		finally:
			for lock in self.locks:
				lock.release()

	def is_ready(self, threshold):
		return threshold <= self.all_ready_count

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
		action, logprob = pa.step(logits_cpu[i])
		actions.append(action)
		logprobs.append(logprob)

	return (
		np.array(actions, dtype=np.int64),
		np.array(logprobs, dtype=np.float32),
		value.cpu().numpy()
	)


def collect_env_data(
	env: Environment,
	env_idx: int,
	model: Policy,
	buffer: TrajectoryBuffer,
	stack_frame: int,
	n_players: int,
	device: torch.device,
	use_threading: bool = True,
	dtype: torch.dtype = torch.float32,
):
	previous_actions = [PreviousAction() for _ in range(n_players)]

	obs, _ = env.reset()
	obs_tensor = torch.tensor(obs, dtype=dtype).permute(0, 3, 1, 2).to(device)
	obs_tensor = obs_tensor.repeat(1, stack_frame, 1, 1)  # (N, C, H, W) 형태로 변환

	while True:
		obs_tensor_temp = torch.tensor(obs, dtype=dtype)
		obs_tensor_temp = obs_tensor_temp.permute(0, 3, 1, 2).to(device)
		obs_tensor[:, :-4, ...] = obs_tensor[:, 4:, ...].clone()
		obs_tensor[:, -4:, ...] = obs_tensor_temp

		actions, logprobs, values = select_action(
			model,
			obs_tensor,
			previous_actions,
		)

		next_obs_batch, rewards, terminateds, truncateds, infos = env.step(actions)

		for p_idx in range(n_players):
			transition = Transition(
				obs=obs[p_idx].copy(),
				action=actions[p_idx],
				log_prob=logprobs[p_idx],
				reward=rewards[p_idx],
				value=values[p_idx],
				done=terminateds[p_idx] or truncateds[p_idx],
			)
			buffer.add_transition(
				env_idx=env_idx,
				player_idx=p_idx,
				transition=transition,
			)

			if transition.done:
				obs[p_idx] = env.reset_player(p_idx)[0]
				temp = torch.tensor(obs[p_idx], dtype=dtype).permute(2, 0, 1).to(device)
				obs_tensor[p_idx, :, :, :] = temp.repeat(stack_frame, 1, 1)
			else:
				obs[p_idx] = next_obs_batch[p_idx]

		if not use_threading:
			break


def train(checkpoint_path=None):
	root_path = Path(__file__).parent
	model_train_time = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	log_folder = root_path / "checkpoints" / model_train_time
	writer = SummaryWriter(log_dir=log_folder.as_posix())

	save_path = root_path / "checkpoints" / model_train_time
	save_path.mkdir(exist_ok=True)

	save_rollout_step = 50  # 50번의 rollout마다 저장
	global_step = 0  # 맨 위에 선언

	n_envs = 1
	n_players = 4  # 플레이어 수
	map_list = ["map1.json", "map2.json", "map3.json", "map4.json"]
	envs = [
		Environment(
			name=f"env_{i}_{map_list[i].split('.')[0]}",
			map_path=root_path / "stage_map/stage/stage1" / map_list[i],
			player_num=n_players,
		) for i in range(n_envs)
	]
	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

	stack_frame = 3
	model = Policy(in_channels=4 * stack_frame).to(device)
	if checkpoint_path is not None and checkpoint_path.exists():
		print(f"Loading model from {checkpoint_path}")
		model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
	buffer = TrajectoryBuffer(
		env_num=n_envs, player_num=n_players, stack_frame=stack_frame,
		gamma=0.997, lam=0.985,
	)

	max_steps_per_update = 2048 // 4

	use_threading = True
	threads = []

	if use_threading:
		threads = [
			Thread(
				target=collect_env_data,
				args=(envs[i], i, model, buffer, stack_frame, n_players, device),
				name=f"collect_env_data_{i}",
			) for i in range(n_envs)
		]
		for thread in threads:
			thread.start()

	rollout_step = 0
	for _ in trange(int(1e6)):
		time.sleep(0.5)  # 잠시 대기하여 CPU 사용량을 줄임
		if not use_threading:
			for i in range(n_envs):
				collect_env_data(envs[i], i, model, buffer, stack_frame, n_players, device, use_threading)
		if buffer.is_ready(max_steps_per_update):
			rollout = buffer.get_rollout(max_steps_per_update)
			train_step(model, optimizer, rollout, device, writer=writer, global_step=rollout_step)
			rollout_step += 1

		if rollout_step != 0 and rollout_step % save_rollout_step == 0:
			torch.save(model.state_dict(), save_path / f"model_step_{global_step}.pt")
			print(f"✅ Model saved at step {global_step}")

	if use_threading:
		for thread in threads:
			thread.join()


def train_step(
	model,
	optimizer,
	rollout: List[Transition],
	device, writer, global_step, epochs=4, minibatch_size=64, dtype=torch.float32
):
	obs = torch.tensor(np.stack([t.obs for t in rollout]), dtype=dtype).permute(0, 3, 1, 2).to(device)
	obs /= 255.0  # Normalize the observations to [0, 1]
	actions = torch.tensor(np.stack([t.action for t in rollout]), dtype=torch.int64).to(device)
	# rewards = torch.tensor(np.stack([t.reward for t in rollout]), dtype=dtype).to(device)
	logprobs = torch.tensor(np.stack([t.log_prob for t in rollout]), dtype=dtype).to(device)
	advantages = torch.tensor(np.stack([t.advantage for t in rollout]), dtype=dtype).to(device)
	returns = torch.tensor(np.stack([t.returns for t in rollout]), dtype=dtype).to(device)
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

	for _ in trange(epochs):
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
