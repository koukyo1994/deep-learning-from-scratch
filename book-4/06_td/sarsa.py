from collections import defaultdict, deque
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


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state: State) -> Action:
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state: State, action: Action, reward: float, done: bool):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        # 次のQ関数
        next_q = 0 if done else self.Q[next_state, next_action]

        # TD法による更新
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 方策改善
        self.pi[state] = greedy_probs(self.Q, state, epsilon=self.epsilon)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaAgent()

    episodes = 10000
    for episode in tqdm(range(episodes)):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)
            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)
