from typing import List
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Model, optimizers


def prepare_image(states: List[np.ndarray]) -> np.ndarray:
    gray_images = []
    for state in states:
        state = state[32:195, :, :]
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84))
        gray = gray / 255.0
        gray_images.append(gray)
    return np.stack(gray_images, axis=2)


class QCNN(Model):
    def __init__(self, action_size: int):
        super().__init__()
        self.conv1 = L.Conv2d(32, kernel_size=8, stride=4)
        self.conv2 = L.Conv2d(64, kernel_size=4, stride=2)
        self.conv3 = L.Conv2d(64, kernel_size=3, stride=1)

        self.linear1 = L.Linear(512)
        self.linear2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 1.0
        self.buffer_size = 1000


if __name__ == "__main__":
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", new_step_api=True)
    states = []
    states.append(env.reset())

    states.append(env.step(5)[0])
    states.append(env.step(5)[0])

    image = prepare_image(states)

    plt.imshow(image)
    plt.show()
