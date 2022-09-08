import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Model, optimizers
from PIL import Image
from tqdm import tqdm


class Policy(Model):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi =  Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]  # バッチの軸
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    episodes = args.episodes
    env = gym.make("CartPole-v1", new_step_api=True)
    agent = Agent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)

    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    plt.plot(reward_history)
    plt.tight_layout()
    plt.savefig(f"./figures/REINFORCE_reward_history_step{episodes}.png")

    agent.epsilon = 0
    state = env.reset()
    done = False
    total_reward = 0
    frames = []

    while not done:
        action, _ = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        img = Image.fromarray(env.render(mode="rgb_array"))
        frames.append(img)
    print("Total Reward: ", total_reward)
    frames[0].save(f"./figures/REINFORCE_cartpole_step{episodes}.gif", save_all=True, append_images=frames[1:])
