from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bandit import Bandit


class SoftmaxAgent:
    def __init__(self, tau: float = 0.01, action_size: int = 10) -> None:
        self.tau = tau
        self.means = np.ones(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action: int, reward: int) -> None:
        self.ns[action] += 1
        self.means[action] += (reward - self.means[action]) / self.ns[action]

    def get_action(self) -> int:
        max_means = (self.means / self.tau).max()
        select_rate = np.exp(self.means / self.tau - max_means) / np.sum(
            np.exp(self.means / self.tau - max_means)
        )
        return np.random.choice(len(select_rate), p=select_rate)


class SoftmaxAnnealingAgent:
    def __init__(
        self,
        initial_tau: float = 0.01,
        k: float = 100.0,
        action_size: int = 10,
    ) -> None:
        self.initial_tau = initial_tau
        self.k = k
        self.means = np.ones(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action: int, reward: int) -> None:
        self.ns[action] += 1
        self.means[action] += (reward - self.means[action]) / self.ns[action]

    def get_action(self) -> int:
        tau = self.initial_tau / np.log(
            self.k * self.ns.sum() + 2
        )
        max_means = (self.means / tau).max()
        select_rate = np.exp(self.means / tau - max_means) / np.sum(
            np.exp(self.means / tau - max_means)
        )
        return np.random.choice(len(select_rate), p=select_rate)


if __name__ == "__main__":
    runs = 200
    steps = 1000
    tau = 0.05
    initial_tau = 0.1
    k = 100.0

    softmax_rates = np.zeros((runs, steps))
    annealed_rates = np.zeros((runs, steps))

    for run in tqdm(range(runs)):
        bandit = Bandit()
        agent = SoftmaxAgent(tau)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        softmax_rates[run] = rates

    for run in tqdm(range(runs)):
        bandit = Bandit()
        agent = SoftmaxAnnealingAgent(initial_tau, k)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))
        annealed_rates[run] = rates

    avg_softmax_rates = np.average(softmax_rates, axis=0)
    avg_annealed_rates = np.average(annealed_rates, axis=0)

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_softmax_rates, label="softmax")
    plt.plot(avg_annealed_rates, label="annealed softmax")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "softmax_200_runs_avg.png")
    plt.close()
