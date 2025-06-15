import re

# 로그 파일 읽기
with open("log.log") as f:
    log_text_list = f.readlines()

# 초기값 설정
final_checkpoint_count = 0
wall_die_count = 0
perfect_checkpoint_count = 0  # 5점 이상 보상
rewards = []

# 각 줄을 순회하며 처리
for line in log_text_list:
    if "final checkpoint" in line:
        match = re.search(r"reward: ([\d.]+)", line)
        if match:
            reward = float(match.group(1))
            rewards.append(reward)
            final_checkpoint_count += 1
            if reward >= 5.0:
                perfect_checkpoint_count += 1
    elif "wall die" in line:
        wall_die_count += 1

# 전체 종료 횟수
total_episodes = final_checkpoint_count + wall_die_count

# 평균 reward 계산
average_reward = sum(rewards) / len(rewards) if rewards else 0

# 비율 계산 (0으로 나누는 것 방지)
def ratio(part, total):
    return (part / total * 100) if total > 0 else 0

# 결과 출력
print(f"총 종료 횟수: {total_episodes}")
print(f"Final checkpoint 횟수: {final_checkpoint_count} ({ratio(final_checkpoint_count, total_episodes):.2f}%)")
print(f"Wall die 횟수: {wall_die_count} ({ratio(wall_die_count, total_episodes):.2f}%)")
print(f"모든 동전을 수집한 Final checkpoint (5점 이상) 횟수: {perfect_checkpoint_count} ({ratio(perfect_checkpoint_count, total_episodes):.2f}%)")
print(f"Final checkpoint 평균 reward: {average_reward:.3f}")
