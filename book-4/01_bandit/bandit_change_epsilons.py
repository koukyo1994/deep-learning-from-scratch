from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bandit import Agent, Bandit


if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilons = [
        0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]

    results = {
        e: np.zeros((runs, steps))
        for e in epsilons
    }
    avg_results = {}

    for epsilon in epsilons:
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
            results[epsilon][run] = rates

        avg_rates = np.average(results[epsilon], axis=0)
        avg_results[epsilon] = avg_rates

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    for epsilon in avg_results:
        plt.plot(avg_results[epsilon], label=f"$\epsilon=${epsilon:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "epsilon_greedy_change_epsilong.png")
    plt.close()
