import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0]for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done
