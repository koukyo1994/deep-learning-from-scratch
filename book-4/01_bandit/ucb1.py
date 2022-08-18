from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bandit import Bandit


class UCB1Agent:
    def __init__(self, R: float = 1.0, action_size: int = 10) -> None:
        self.R = R
        self.ones = np.zeros(action_size)
        self.zeros = np.zeros(action_size)

    def update(self, action: int, reward: int) -> None:
        if reward == 0:
            self.zeros[action] += 1
        elif reward == 1:
            self.ones[action] += 1
        else:
            raise ValueError

    def get_action(self) -> int:
        eps = 1e-8
        scores = (
            self.ones / (self.ones + self.zeros + eps) +
            self.R * (
                2.0 * np.log1p((self.ones + self.zeros).sum()) /
                (self.ones + self.zeros + eps)
            ) ** 0.5
        )
        return np.argmax(scores)


if __name__ == "__main__":
    runs = 200
    steps = 1000
    R = 1.0

    all_rates = np.zeros((runs, steps))

    for run in tqdm(range(runs)):
        bandit = Bandit()
        agent = UCB1Agent(R)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_rates)
    plt.tight_layout()
    plt.savefig(fig_dir / "ucb1_200_runs_avg.png")
    plt.close()
