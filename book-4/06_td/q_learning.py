from collections import defaultdict
from typing import Tuple

import numpy as np
from tqdm import tqdm

from gridworld import GridWorld
from sarsa import greedy_probs


State = Tuple[int, int]
Action = int
Prob = float


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state: State) -> Action:
        # 挙動方策bを用いる
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        # 次の状態で価値が最大となる行動を取った時の行動価値関数
        if done:
            max_q = 0
        else:
            next_qs = [self.Q[next_state, action] for action in range(self.action_size)]
            max_q = max(next_qs)

        # TD法による更新
        target = reward + self.gamma * max_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 方策改善
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0.0)
        self.b[state] = greedy_probs(self.Q, state, epsilon=self.epsilon)


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
