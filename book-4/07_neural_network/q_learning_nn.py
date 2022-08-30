import argparse
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Model, optimizers
from tqdm import tqdm

from gridworld import GridWorld

State = Tuple[int, int]
Action = int


def one_hot(state: State, height: int, width: int) -> np.ndarray:
    vec = np.zeros(height * width, dtype=np.float32)
    y, x = state
    idx = width * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]  # バッチのための新しい軸を追加


class QNet(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y


class QLearningAgent:
    def __init__(self, hidden_size: int = 100):
        self.gamma = 0.9
        self.lr = 0.001
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet(hidden_size, out_size=self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state: np.ndarray) -> Action:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state: np.ndarray, action: Action, reward: float, next_state: np.ndarray, done: bool):
        if done:
            max_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            max_q = next_qs.max(axis=1)
            max_q.unchain()

        target = reward + self.gamma * max_q
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()
        return loss.data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_map", action="store_true")
    parser.add_argument("--episodes", default=1000, type=int)
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
    agent = QLearningAgent(100)

    height, width = env.reward_map.shape

    episodes = args.episodes
    loss_history = []
    Q = defaultdict(lambda: 0)

    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = one_hot(state, height, width)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state, height, width)

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        average_loss = total_loss / cnt
        loss_history.append(average_loss)

    # 描画用
    for state in env.states():
        state_oh = one_hot(state, height, width)
        q = agent.qnet(state_oh).data[0]
        for i in range(agent.action_size):
            Q[state, i] = q[i]

    # 損失の描画
    plt.ylabel("loss")
    plt.xlabel("episode")
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.tight_layout()
    figtitle = f"loss_history_ep{args.episodes}"
    if args.original_map:
        figtitle += "_original_map.png"
    else:
        figtitle += ".png"
    plt.savefig(Path("./figures") / figtitle)

    env.render_q(Q)
