import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from stable_baselines3.a2c.policies import ActorCriticPolicy
from stable_baselines3.common.logger import configure
from pathlib import Path
from datetime import datetime
import json
from copy import deepcopy
import cv2
import torch

# 미로 탈출 환경 import
from stage_map.env import Environment, SingleEnvironment


class SingleAgentWrapper(gym.Env):
	"""
	멀티 에이전트 환경을 단일 에이전트 환경으로 래핑하는 클래스
	"""

	def __init__(
		self,
		map_path: Path = None,
		player_id: int = 0,
		fps: int = 30,
		max_time_seconds: int = 60,
		train_mode: bool = True
	):
		super(SingleAgentWrapper, self).__init__()

		self.player_id = player_id
		self.train_mode = train_mode
		self.init_first_reset = False
		self.map_path = map_path

		with open(self.map_path.as_posix(), 'r') as f:
			self.map_data_dict = json.load(f)
			self.map_data_dict_temp = deepcopy(self.map_data_dict)
			self.randomize_map()
		self.env = SingleEnvironment(
			name="train",
			# map_path=self.map_path,
			map_data_dict=self.map_data_dict_temp,
			fps=fps,
			max_time_seconds=max_time_seconds,
			train_mode=train_mode
		)

		# 액션 공간: 9개 방향 (정지, 상, 우상, 우, 우하, 하, 좌하, 좌, 좌상)
		self.action_space = gym.spaces.Discrete(9)

		# 관찰 공간: 4채널 이미지 (C, H, W) 형태
		self.observation_space = self.env.observation_space

	def reset(self, seed=None, options=None):
		"""환경 초기화"""
		self.randomize_map()
		self.env.reset_map(self.map_data_dict_temp)
		obs, info = self.env.reset()
		return obs, info

	def step(self, action):
		"""환경 한 스텝 진행"""
		# 액션을 배열로 변환 (단일 플레이어이므로 크기 1)
		actions = np.array([action])
		obs, reward, terminated, truncated, infos = self.env.step(actions)
		return obs, reward, terminated, truncated, infos

	def render(self, mode="rgb_array"):
		"""렌더링"""
		if mode == "rgb_array":
			# 원본 환경의 렌더링 결과를 RGB 배열로 반환
			image = self.env.render(mode="high_rgb_array")
			return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		else:
			return self.env.render(mode=mode)

	def close(self):
		"""환경 종료"""
		self.env.close()

	def randomize_map(self):
		"""
		맵 데이터를 무작위로 변경합니다.
		"""
		temp = deepcopy(self.map_data_dict)
		# matrix = np.array(temp["matrix"])
		# ground_mask = matrix == 1

		# 설정
		coin_count = int(np.clip(np.random.normal(loc=5, scale=1), 1, 20))  # 최소 1, 최대 20개
		coin_size = 50
		coin_radius = coin_size / 2

		coin_x1, coin_y1, coin_x2, coin_y2 = 300, 300, 800, 500

		# 배치 가능한 범위 (반지름 고려)
		xmin = coin_x1 + coin_radius
		xmax = coin_x2 - coin_radius
		ymin = coin_y1 + coin_radius
		ymax = coin_y2 - coin_radius

		positions = []

		max_attempts = 1000
		attempt = 0

		while len(positions) < coin_count and attempt < max_attempts:
			attempt += 1
			x = round(np.random.uniform(xmin, xmax) / 5) * 5  # 5의 배수로 반올림
			y = round(np.random.uniform(ymin, ymax) / 5) * 5  # 5의 배수로 반올림

			# 겹치는지 검사
			overlap = False
			for (px, py) in positions:
				dist = np.hypot(px - x, py - y)
				if dist < coin_size:  # 반지름 * 2
					overlap = True
					break

			if not overlap:
				positions.append((round(x), round(y)))

		if len(positions) < coin_count:
			print(f"⚠️ {coin_count}개 중 {len(positions)}개만 배치됨 (공간 부족)")

		temp["coin_list"] = [
			{
				"pos": [pos[0], pos[1]],
			}
			for pos in positions
		]

		self.map_data_dict_temp = temp


class TrainingCallback(BaseCallback):
	"""
	훈련 중 성능을 모니터링하는 콜백
	"""

	def __init__(self, eval_freq: int = 1000, verbose: int = 1, model_save_folder=None):
		super(TrainingCallback, self).__init__(verbose)
		self.eval_freq = eval_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.save_freq = 100000
		self.model_save_folder = model_save_folder

	def _on_step(self) -> bool:
		# 에피소드가 끝났을 때 정보 수집
		if self.locals.get('dones') and self.locals['dones'][0]:
			if 'episode' in self.locals['infos'][0]:
				episode_info = self.locals['infos'][0]['episode']
				self.episode_rewards.append(episode_info['r'])
				self.episode_lengths.append(episode_info['l'])

				if len(self.episode_rewards) % 10 == 0:
					recent_rewards = self.episode_rewards[-10:]
					print(f"Episode {len(self.episode_rewards)}: "
						  f"Recent avg reward: {np.mean(recent_rewards):.2f}")
		if self.n_calls % self.save_freq == 0:
			model_path = Path(self.model_save_folder) / f"model_{self.n_calls}.pth"
			torch.save(self.model.policy.state_dict(), model_path)
			if self.verbose > 0:
				print(f"✅ 모델 저장됨: {model_path}")
		return True


def create_maze_env(map_path: Path = None, **kwargs):
	"""미로 환경 생성 함수"""

	def _init():
		return SingleAgentWrapper(map_path=map_path, **kwargs)

	return _init


def demonstrate_maze_agent(model, env, episodes=3):
	"""미로 탈출 에이전트 시연"""
	if model is None or env is None:
		print("모델 또는 환경이 없어 시연을 건너뜁니다.")
		return

	print(f"\n=== 미로 탈출 에이전트 시연 ({episodes}에피소드) ===")

	for episode in range(episodes):
		obs = env.reset()
		total_reward = 0
		step_count = 0

		print(f"\n에피소드 {episode + 1}:")

		while True:
			# 모델로 액션 예측
			action, _states = model.predict(obs, deterministic=True)

			# 환경에서 액션 실행
			obs, reward, done, info = env.step(action)
			total_reward += reward[0] if hasattr(reward, '__len__') else reward
			step_count += 1

			# 종료 조건 확인
			if done[0] if hasattr(done, '__len__') else done:
				break

			if step_count > 1000:  # 무한 루프 방지
				break

		print(f"  총 보상: {total_reward:.2f}, 스텝 수: {step_count}")


def visualize_agent_performance(model, env, save_video=False):
	"""에이전트 성능 시각화"""
	if model is None or env is None:
		print("모델 또는 환경이 없어 시각화를 건너뜁니다.")
		return

	print("\n=== 에이전트 성능 시각화 ===")
	print("ESC 키를 눌러 종료하세요.")

	obs = env.reset()
	total_reward = 0
	step_count = 0

	# 단일 환경에서 원본 환경에 접근
	original_env = env.envs[0].env

	while True:
		# 모델로 액션 예측
		action, _states = model.predict(obs, deterministic=True)

		# 환경에서 액션 실행
		obs, reward, done, info = env.step(action)
		total_reward += reward[0] if hasattr(reward, '__len__') else reward
		step_count += 1

		# 렌더링
		try:
			image = original_env.render(mode="high_rgb_array")
			if image is not None:
				# BGR to RGB 변환
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				# 정보 표시
				cv2.putText(image_rgb, f"Step: {step_count}", (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				cv2.putText(image_rgb, f"Reward: {total_reward:.2f}", (10, 70),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				cv2.putText(image_rgb, f"Action: {action[0] if hasattr(action, '__len__') else action}", (10, 110),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

				cv2.imshow("Maze Agent", image_rgb)

				key = cv2.waitKey(50)  # 50ms 대기
				if key == 27:  # ESC 키
					break

		except Exception as e:
			print(f"렌더링 오류: {e}")

		# 에피소드 종료 확인
		if done[0] if hasattr(done, '__len__') else done:
			print(f"에피소드 종료 - 총 보상: {total_reward:.2f}, 스텝 수: {step_count}")
			obs = env.reset()
			total_reward = 0
			step_count = 0

		if step_count > 2000:  # 무한 루프 방지
			print("최대 스텝 수 도달")
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	root_path = Path(__file__).parent
	model_train_time = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	log_folder = root_path / "checkpoints" / model_train_time

	try:
		print("=== 미로 탈출 환경에서 A2C 테스트 ===")

		# 환경 생성
		map_path = Path("stage_map/stage/stage2/make.json")
		if not map_path.exists():
			map_path = None
			print("맵 파일을 찾을 수 없어 기본 설정을 사용합니다.")

		# 벡터화된 환경 생성 (단일 환경)
		env = DummyVecEnv([create_maze_env(
			map_path=map_path,
			train_mode=True,
			max_time_seconds=30
		)])

		# 프레임 쌓기
		env = VecFrameStack(env, n_stack=3, channels_order='first')  # 3채널 스택 (이미지 입력을 위한)

		# A2C 모델 생성 (이미지 입력을 위한 CNN 정책 사용)
		model = A2C(
			"CnnPolicy",  # CNN 정책 (이미지 입력용)
			env,
			verbose=1,
			learning_rate=1e-5,
			n_steps=64,  # 더 많은 스텝 수집
			gamma=0.99,
			gae_lambda=0.95,
			ent_coef=0.01,  # 탐험 장려
			vf_coef=0.5,
			max_grad_norm=0.5,
			use_rms_prop=True,  # RMSprop 옵티마이저 사용
			rms_prop_eps=1e-5,
			# device="cuda" if torch.cuda.is_available() else (
			# 	"mps" if torch.backends.mps.is_available() else "cpu"
			# ),
		)
		new_logger = configure(log_folder.as_posix(), ["stdout", "tensorboard"])
		model.set_logger(new_logger)
		device = "cuda" if torch.cuda.is_available() else (
			"mps" if torch.backends.mps.is_available() else "cpu"
		)
		# model_path = Path("a2c_maze_model.zip")
		# if model_path.exists():
		# 	print(f"기존 모델을 로드합니다: {model_path}")
		# 	model = A2C.load(model_path, env=env)
		model_path = Path("bc_actor_weights.pth")
		if model_path.exists():
			model.policy.load_state_dict(torch.load(model_path, map_location=device))

		# 콜백 설정
		callback = TrainingCallback(
			eval_freq=1000,
			model_save_folder=log_folder,
		)

		# 모델 훈련
		print("\n--- 훈련 시작 ---")
		print("이미지 기반 환경이므로 훈련에 시간이 걸릴 수 있습니다...")

		model.learn(total_timesteps=10000000, callback=callback)

		if model is not None and env is not None:
			# 2. 훈련된 에이전트 시연
			# demonstrate_maze_agent(maze_model, maze_env)

			# 4. 시각화 (선택사항)
			print("\n시각화를 실행하시겠습니까? (y/n): ", end="")
			try:
				user_input = input().lower()
				if user_input == 'y':
					visualize_agent_performance(model, env)
			except:
				print("입력을 받을 수 없어 시각화를 건너뜁니다.")

		print("\n" + "=" * 60)
		print("미로 탈출 A2C 테스트가 완료되었습니다!")

	except KeyboardInterrupt:
		print("\n사용자에 의해 중단되었습니다.")
	except Exception as e:
		print(f"예상치 못한 오류가 발생했습니다: {e}")
		import traceback

		traceback.print_exc()