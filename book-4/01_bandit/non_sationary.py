from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bandit import Agent


class NonStatBandit:
    def __init__(self, arms: int = 10) -> None:
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)  # ノイズを追加
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, actions: int = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action: int, reward: int) -> None:
        self.Qs[action] += self.alpha * (reward - self.Qs[action])

    def get_action(self) -> int:
        if self.epsilon > np.random.rand():
            # 探索
            return np.random.randint(0, len(self.Qs))
        # 活用
        return np.argmax(self.Qs)


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8

    sample_avg_rates = np.zeros((runs, steps))
    alpha_avg_rates = np.zeros((runs, steps))

    # 標本平均を使うAgent
    for run in range(runs):
        bandit = NonStatBandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        sample_avg_rates[run] = rates

    # AlphaAgent
    for run in range(runs):
        bandit = NonStatBandit()
        agent = AlphaAgent(epsilon, alpha)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        alpha_avg_rates[run] = rates

    avg_rates_sample = np.average(sample_avg_rates, axis=0)
    avg_rates_alpha = np.average(alpha_avg_rates, axis=0)

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_rates_sample, label="sample average agent")
    plt.plot(avg_rates_alpha, label=f"alpha={alpha:.2f} agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "nonstationary_problem.png")
    plt.close()
