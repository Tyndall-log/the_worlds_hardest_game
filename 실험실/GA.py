import numpy as np
import pandas as pd

# 감가율
gamma = 0.99

# 보상 리스트
rewards = [0, 0, 2, 0, -1, 0, 0, 0, 1, 2]

# critic이 예측한 값
v_preds = [0.5, 0.4, 1.8, 0.6, -0.5, 0.3, 0.2, 0.1, 1+2*gamma, 2]

# G_t 계산 함수
def compute_returns(rewards, gamma):
	G = []
	for t in range(len(rewards)):
		g = 0
		for k in range(t, len(rewards)):
			g += (gamma ** (k - t)) * rewards[k]
		G.append(round(g, 4))
	return G

# 계산
returns = compute_returns(rewards, gamma)
advantages = [round(G - V, 4) for G, V in zip(returns, v_preds)]
critic_losses = [round((G - V) ** 2, 4) for G, V in zip(returns, v_preds)]

# 정리
df = pd.DataFrame({
	"Step": list(range(len(rewards))),
	"Reward (r)": rewards,
	"Pred V(s)": v_preds,
	"Return Gₜ": returns,
	"Advantage Aₜ": advantages,
	"Critic Loss": critic_losses
})

# 출력
print(df)

