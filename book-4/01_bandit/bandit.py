from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, arms: int = 10) -> None:
        self.rates = np.random.rand(arms)  # 各マシンの確率

    def play(self, arm: int) -> int:
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action: int, reward: int) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            # 探索
            return np.random.randint(0, len(self.Qs))
        # 活用
        return np.argmax(self.Qs)


if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(total_reward)

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.tight_layout()
    plt.savefig(fig_dir / "epsilon_greedy_rewards.png")
    plt.close()

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.tight_layout()
    plt.savefig(fig_dir / "epsilon_greedy_rates.png")
    plt.close()
