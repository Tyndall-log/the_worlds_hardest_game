import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import torch

# 미로 탈출 환경 import
from stage_map.env import Environment


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

		# 원본 환경 생성 (단일 플레이어)
		self.env = Environment(
			name=f"maze_env_{player_id}",
			map_path=map_path if map_path else Path(__file__).parent / "stage_map/stage/stage0/map1.json",
			player_num=1,  # 단일 플레이어
			fps=fps,
			max_time_seconds=max_time_seconds,
			train_mode=train_mode
		)

		# 액션 공간: 9개 방향 (정지, 상, 우상, 우, 우하, 하, 좌하, 좌, 좌상)
		self.action_space = gym.spaces.Discrete(9)

		# 관찰 공간: 4채널 이미지 (C, H, W) 형태로 변환
		original_shape = self.env.observation_space.shape  # (1, 4, H, W)
		self.observation_space = gym.spaces.Box(
			low=0,
			high=255,
			shape=(original_shape[1], original_shape[2], original_shape[3]),  # (C, H, W)
			dtype=np.uint8
		)

	def reset(self, seed=None, options=None):
		"""환경 초기화"""
		if self.init_first_reset is False:
			obs, info = self.env.reset(seed=seed, options=options)
			# 관찰값 형태 변환: (1, H, W, 4) -> (4, H, W)
			obs_single = obs[0].transpose(2, 0, 1)
			self.init_first_reset = True
		else:
			obs, info = self.env.reset_player(self.player_id)
			obs_single = obs.transpose(2, 0, 1)  # (H, W, 4) -> (4, H, W)
		return obs_single, info

	def step(self, action):
		"""환경 한 스텝 진행"""
		# 액션을 배열로 변환 (단일 플레이어이므로 크기 1)
		actions = np.array([action])

		obs, rewards, terminated, truncated, infos = self.env.step(actions)

		# 관찰값 형태 변환: (1, H, W, 4) -> (4, H, W)
		obs_single = obs[0].transpose(2, 0, 1)

		# 단일 값으로 변환
		reward = rewards[0]
		done = terminated[0]
		trunc = truncated[0]

		if done or trunc:
			# # 에피소드가 끝났을 때 리셋
			# obs_single, _ = self.env.reset()
			pass

		return obs_single, reward, done, trunc, infos

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


class TrainingCallback(BaseCallback):
	"""
	훈련 중 성능을 모니터링하는 콜백
	"""

	def __init__(self, eval_freq: int = 1000, verbose: int = 1):
		super(TrainingCallback, self).__init__(verbose)
		self.eval_freq = eval_freq
		self.episode_rewards = []
		self.episode_lengths = []

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

		return True


def create_maze_env(map_path: Path = None, **kwargs):
	"""미로 환경 생성 함수"""

	def _init():
		return SingleAgentWrapper(map_path=map_path, **kwargs)

	return _init


def test_a2c_with_maze():
	"""미로 탈출 환경에서 A2C 테스트"""
	print("=== 미로 탈출 환경에서 A2C 테스트 ===")

	try:
		# 환경 생성
		map_path = Path("stage_map/stage/stage0/map1.json")
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
			learning_rate=0.0003,  # 이미지 환경에 적합한 학습률
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
		model_path = Path("a2c_maze_model.zip")
		if model_path.exists():
			print(f"기존 모델을 로드합니다: {model_path}")
			model = A2C.load(model_path, env=env)

		# 훈련 전 성능 평가
		print("\n--- 훈련 전 성능 ---")
		try:
			mean_reward_before, std_reward_before = evaluate_policy(
				model, env, n_eval_episodes=3, deterministic=True
			)
			print(f"평균 보상: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
		except Exception as e:
			print(f"초기 평가 중 오류: {e}")
			mean_reward_before = -999

		# 콜백 설정
		callback = TrainingCallback(eval_freq=1000)

		# 모델 훈련
		print("\n--- 훈련 시작 ---")
		print("이미지 기반 환경이므로 훈련에 시간이 걸릴 수 있습니다...")

		model.learn(total_timesteps=100000, callback=callback)

		# 훈련 후 성능 평가
		print("\n--- 훈련 후 성능 ---")
		try:
			mean_reward_after, std_reward_after = evaluate_policy(
				model, env, n_eval_episodes=5, deterministic=True
			)
			print(f"평균 보상: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

			# 성능 향상 확인
			if mean_reward_before != -999:
				improvement = mean_reward_after - mean_reward_before
				print(f"성능 향상: {improvement:.2f}")

		except Exception as e:
			print(f"최종 평가 중 오류: {e}")

		return model, env

	except ImportError as e:
		print(f"미로 환경을 불러올 수 없습니다: {e}")
		print("stage_map.env 모듈이 올바른 위치에 있는지 확인하세요.")
		return None, None
	except Exception as e:
		print(f"환경 생성 중 오류가 발생했습니다: {e}")
		return None, None


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


def save_and_load_maze_model(model):
	"""미로 모델 저장 및 로드 테스트"""
	if model is None:
		print("모델이 없어 저장/로드를 건너뜁니다.")
		return None

	print("\n=== 미로 모델 저장 및 로드 테스트 ===")

	try:
		# 모델 저장
		model.save("a2c_maze_model")
		print("모델이 저장되었습니다: a2c_maze_model.zip")

		# 모델 로드
		loaded_model = A2C.load("a2c_maze_model")
		print("모델이 로드되었습니다.")

		return loaded_model

	except Exception as e:
		print(f"모델 저장/로드 중 오류: {e}")
		return None


if __name__ == "__main__":
	# 필요한 패키지 설치 안내
	print("다음 패키지들이 필요합니다:")
	print("pip install stable-baselines3[extra] gymnasium opencv-python matplotlib")
	print("\n" + "=" * 60)

	try:
		if Path("a2c_maze_model").exists():
			A2C.load("a2c_maze_model")

		# 1. 미로 탈출 환경에서 A2C 테스트
		maze_model, maze_env = test_a2c_with_maze()

		if maze_model is not None and maze_env is not None:
			# 2. 훈련된 에이전트 시연
			demonstrate_maze_agent(maze_model, maze_env)

			# 3. 모델 저장/로드 테스트
			loaded_model = save_and_load_maze_model(maze_model)

			# 4. 시각화 (선택사항)
			print("\n시각화를 실행하시겠습니까? (y/n): ", end="")
			try:
				user_input = input().lower()
				if user_input == 'y':
					visualize_agent_performance(loaded_model or maze_model, maze_env)
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