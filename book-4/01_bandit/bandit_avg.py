from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bandit import Agent, Bandit


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilon = 0.1

    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
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
    plt.savefig(fig_dir / "epsilon_greedy_200_runs_avg.png")
    plt.close()
