import numpy as np

def generate_inertia_reward_matrix(speed=1.0, mass=1.0):
	# 9개의 방향 (정지 포함)
	s = speed
	movement = [
		(0, 0),   # 0: 정지
		(0, -s),  # 1: ↑
		(s, -s),  # 2: ↗
		(s, 0),   # 3: →
		(s, s),   # 4: ↘
		(0, s),   # 5: ↓
		(-s, s),  # 6: ↙
		(-s, 0),  # 7: ←
		(-s, -s), # 8: ↖
	]

	reward_matrix = np.zeros((9, 9))

	for prev in range(9):
		for curr in range(9):
			prev_vec = np.array(movement[prev])
			curr_vec = np.array(movement[curr])
			delta_v = curr_vec - prev_vec
			energy = 0.5 * mass * np.dot(delta_v, delta_v)
			reward_matrix[prev][curr] = -energy  # 음의 보상

	return reward_matrix

def print_reward_matrix(matrix, title="Reward Matrix"):
	head = [".", "↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]
	print(f"\n{title}")
	print("\t" + "\t".join(f"{h:6}" for h in head))
	for i, row in enumerate(matrix):
		print(f"{head[i]}\t" + "\t".join(f"{val:6.3f}" for val in row))

# 출력
matrix = generate_inertia_reward_matrix(speed=6.0, mass=0.001)
np.set_printoptions(precision=2, suppress=True)
print_reward_matrix(matrix, title="Inertia Reward Matrix (Speed=6.0, Mass=0.001)")

import numpy as np

def generate_inertia_reward_matrix(speed=1.0, penalty_weight=1.0, norm_type="l2"):
	# movement vectors: index 0~8
	s = speed
	movement = [
		(0, 0),   # 0: 정지
		(0, -s),  # 1: ↑
		(s, -s),  # 2: ↗
		(s, 0),   # 3: →
		(s, s),   # 4: ↘
		(0, s),   # 5: ↓
		(-s, s),  # 6: ↙
		(-s, 0),  # 7: ←
		(-s, -s), # 8: ↖
	]

	matrix = np.zeros((9, 9))

	for i, prev in enumerate(movement):
		for j, curr in enumerate(movement):
			delta_v = np.array(curr) - np.array(prev)
			if norm_type == "l1":
				norm = np.sum(np.abs(delta_v))
			elif norm_type == "l2":
				norm = np.linalg.norm(delta_v)
			else:
				raise ValueError("Invalid norm_type. Use 'l1' or 'l2'.")
			matrix[i][j] = -penalty_weight * norm

	return matrix


l1_matrix = generate_inertia_reward_matrix(speed=6.0, penalty_weight=0.01, norm_type="l1")
l2_matrix = generate_inertia_reward_matrix(speed=6.0, penalty_weight=0.01, norm_type="l2")

print_reward_matrix(l1_matrix, "L1 Norm Reward Matrix")
print_reward_matrix(l2_matrix, "L2 Norm Reward Matrix")