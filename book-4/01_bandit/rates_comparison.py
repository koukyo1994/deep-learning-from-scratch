from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bandit import Agent, Bandit
from bayesian_ucb import BayesianUCBAgent
from softmax import SoftmaxAgent, SoftmaxAnnealingAgent
from thompson_sampling import ThompsonSamplingAgent
from ucb1 import UCB1Agent


if __name__ == "__main__":
    runs = 200
    steps = 1000
    algorithms = {
        "epsilon-greedy": {
            "cls": Agent, "params": {"epsilon": 0.1},
        },
        "bayesian-ucb": {
            "cls": BayesianUCBAgent, "params": {"percentile": 0.95}
        },
        "ucb1": {
            "cls": UCB1Agent, "params": {"R": 1.0}
        },
        "softmax": {
            "cls": SoftmaxAgent, "params": {"tau": 0.05}
        },
        "annealed-softmax": {
            "cls": SoftmaxAnnealingAgent, "params": {"initial_tau": 0.01, "k": 100.0}
        },
        "thompson-sampling": {
            "cls": ThompsonSamplingAgent, "params": {}
        },
    }

    results = {
        name: np.zeros((runs, steps))
        for name in algorithms
    }
    avg_results = {}

    for name in algorithms:
        for run in tqdm(range(runs), desc=name):
            bandit = Bandit()
            agent = algorithms[name]["cls"](
                **algorithms[name]["params"]
            )
            total_reward = 0
            rates = []

            for step in range(steps):
                try:
                    action = agent.get_action()
                except ValueError:
                    import pdb

                    pdb.set_trace()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))
            results[name][run] = rates

        avg_rates = np.average(results[name], axis=0)
        avg_results[name] = avg_rates

    fig_dir = Path("./figures")
    fig_dir.mkdir(exist_ok=True, parents=True)

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    for name in avg_results:
        plt.plot(avg_results[name], label=name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rates_comparison_between_algorithms.png")
    plt.close()
