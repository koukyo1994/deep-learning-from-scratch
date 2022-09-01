import copy

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Model, optimizers
from PIL import Image
from tqdm import tqdm

from replay_buffer import ReplayBuffer


class QNet(Model):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(
            self.buffer_size, self.batch_size
        )
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


if __name__ == "__main__":
    episodes = 300
    sync_interval = 10
    env = gym.make("CartPole-v1", new_step_api=True)
    agent = DQNAgent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)

    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    plt.plot(reward_history)
    plt.tight_layout()
    plt.savefig("./figures/dqn_reward_history.png")

    agent.epsilon = 0
    state = env.reset()
    done = False
    total_reward = 0
    frames = []

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        img = Image.fromarray(env.render(mode="rgb_array"))
        frames.append(img)
    print("Total Reward: ", total_reward)
    frames[0].save("./figures/cartpole.gif", save_all=True, append_images=frames[1:])
