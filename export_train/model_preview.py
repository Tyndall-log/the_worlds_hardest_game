from pathlib import Path
import json
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from stable_baselines3.common.env_util import make_vec_env
from imitation.data.types import Trajectory
from torch.utils.data import Dataset
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.a2c import CnnPolicy
from stable_baselines3.sac import CnnPolicy
import torch
import gymnasium as gym
import numpy as np

from stage_map.env import Environment, SingleEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

def load_transitions_from_json(json_path: Path, env_factory) -> Transitions:
	with open(json_path, "r") as f:
		data = json.load(f)

	obs_list = []
	next_obs_list = []
	act_list = []
	done_list = []
	info_list = []

	for traj in data:
		env = env_factory(map_data=traj["map_data"])
		obs = env.reset()[0]  # (batch=1,) 벡터화된 첫 obs
		for i, action in enumerate(traj["actions"]):
			obs_list.append(obs)
			act_list.append(action)

			obs_, _, done, _ = env.step([action])
			obs_ = obs_[0]
			done = done[0]
			next_obs_list.append(obs_)
			# 마지막 행동이거나 중간에 done이면 True
			is_done = (i == len(traj["actions"]) - 1) or done
			done_list.append(is_done)
			info_list.append({})

			if done:
				break
			obs = obs_

	# 마지막 obs가 빠질 수 있으므로 보정
	if len(next_obs_list) < len(obs_list):
		next_obs_list.append(next_obs_list[-1])
		done_list[-1] = True

	# 배열로 변환
	obs_array = np.stack(obs_list)
	next_obs_array = np.stack(next_obs_list)
	acts_array = np.array(act_list)
	dones_array = np.array(done_list)
	infos_array = np.array(info_list)

	return Transitions(
		obs=obs_array,
		acts=acts_array,
		next_obs=next_obs_array,
		dones=dones_array,
		infos=infos_array,
	)

def env_factory(map_data):
	env = SingleEnvironment(
		name="bc_env",
		map_data_dict=map_data,
		fps=30,
	)
	env = DummyVecEnv([lambda: env])  # 벡터화된 환경으로 래핑
	env = VecFrameStack(env, n_stack=3, channels_order="first")  # 프레임 스택
	return env

if __name__ == "__main__":
	root_path = Path(__file__).parent.parent

	transitions = load_transitions_from_json(
		root_path / "export_train/raw_data/20250615_031807(삭제금지).json",
		env_factory=env_factory,
	)

	# 첫 번째 sample로 observation_space, action_space 추출
	obs, act = transitions.obs[0], transitions.acts[0]
	observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=obs.dtype)
	action_space = gym.spaces.Discrete(9)

	policy = CnnPolicy(
		observation_space=observation_space,
		action_space=action_space,
		lr_schedule=lambda _: 1e-3,
	)

	# BC 학습 객체 생성
	bc = BC(
		policy=policy,
		observation_space=observation_space,
		action_space=action_space,
		demonstrations=transitions,
		batch_size=32,
		rng=np.random.default_rng(),
	)

	# 학습
	bc.train(n_epochs=4)

	# 저장
	torch.save(bc.policy.state_dict(), "bc_asc_weights.pth")