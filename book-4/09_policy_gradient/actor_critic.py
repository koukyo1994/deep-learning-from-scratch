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


class Value(Model):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = Policy(self.action_size)
        self.V = Value()
        self.optimizer_pi = optimizers.Adam(
            self.lr_pi
        ).setup(self.pi)
        self.optimizer_v = optimizers.Adam(
            self.lr_v
        ).setup(self.V)

    def get_action(self, state):
        state = state[np.newaxis, :]  # バッチの軸
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # バッチ軸の追加
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # Vの損失
        target = reward + self.gamma * self.V(next_state) * (1 - done)
        target.unchain()
        V = self.V(state)
        loss_v = F.mean_squared_error(V, target)

        # piの損失
        delta = target - V
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.V.cleargrads()
        self.pi.cleargrads()

        loss_v.backward()
        loss_pi.backward()

        self.optimizer_v.update()
        self.optimizer_pi.update()



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

            agent.update(state, prob, reward, next_state, done)
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    plt.plot(reward_history)
    plt.tight_layout()
    plt.savefig(f"./figures/Actor_Critic_reward_history_step{episodes}.png")

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
    frames[0].save(f"./figures/Actor_Critic_cartpole_step{episodes}.gif", save_all=True, append_images=frames[1:])
