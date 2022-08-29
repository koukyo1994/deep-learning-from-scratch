from collections import defaultdict
from typing import Tuple

import numpy as np
from tqdm import tqdm

from gridworld import GridWorld


State = Tuple[int, int]
Action = int
Prob = float


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        self.Q = defaultdict(lambda: 0)

    def get_action(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        if done:
            max_q = 0
        else:
            next_qs = [self.Q[next_state, action] for action in range(self.action_size)]
            max_q = max(next_qs)

        # TD法による更新
        target = reward + self.gamma * max_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in tqdm(range(episodes)):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    env.render_q(agent.Q)
