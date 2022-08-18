from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm import tqdm

from bandit import Bandit



class ThompsonSamplingAgent:
    def __init__(self, action_size: int = 10) -> None:
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
        return np.argmax(
            scipy.stats.beta.rvs(self.ones + 1, self.zeros + 1)
        )


if __name__ == "__main__":
    runs = 200
    steps = 1000

    all_rates = np.zeros((runs, steps))

    for run in tqdm(range(runs)):
        bandit = Bandit()
        agent = ThompsonSamplingAgent()
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
    plt.savefig(fig_dir / "thompson_sampling_200_runs_avg.png")
    plt.close()
