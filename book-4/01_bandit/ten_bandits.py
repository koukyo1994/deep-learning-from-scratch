from pathlib import Path

import matplotlib.pyplot as plt

from bandit import Agent, Bandit


if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    trial_results = []
    for i in range(10):
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
        trial_results.append(rates)

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    for i in range(10):
        plt.plot(trial_results[i])
    plt.tight_layout()
    plt.savefig(fig_dir / "epsilon_greedy_10_trials.png")
    plt.close()
