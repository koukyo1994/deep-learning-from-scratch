import argparse
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple

import numpy as np
from tqdm import tqdm

from gridworld import GridWorld


State = Tuple[int, int]
Action = int
Prob = float


def greedy_probs(
    Q: DefaultDict[Tuple[State, Action], float],
    state: State,
    epsilon: float = 0,
    action_size: int = 4
) -> Dict[Action, Prob]:
    qs = [Q[state, action] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1  # ε-greedyのε
        self.alpha = 0.1  # Q値を更新する際の固定値alpha
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state: State) -> int:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state: State, action: int, reward: float) -> None:
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def update(self) -> None:
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, epsilon=self.epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_map", action="store_true")
    parser.add_argument("--episodes", default=10000, type=int)
    args = parser.parse_args()

    env = GridWorld()
    if args.original_map:
        env.reward_map = np.array([
            [0, 0, 0, 0, 0, 0, -1.0],
            [0, 0, None, 0, None, None, 0],
            [0, 0, None, 0, 1.0, 0, None],
            [0, 0, None, None, None, None, None]
        ])
        env.goal_state = (2, 4)
        env.wall_state = [
            (1, 2), (1, 4), (1, 5),
            (2, 2), (2, 6),
            (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)
        ]
        env.start_state = (3, 0)
    agent = McAgent()

    episodes = args.episodes
    for episode in tqdm(range(episodes)):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break
            state = next_state
    env.render_q(agent.Q)
