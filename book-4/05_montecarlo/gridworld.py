from typing import Tuple, Optional

import numpy as np

from _renderer import Renderer


class GridWorld:
    def __init__(self) -> None:
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array([
            [0, 0, 0, 1.0],
            [0, None, 0, -1.0],
            [0, 0, 0, 0]
        ])

        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return self.reward_map.shape[1]

    @property
    def shape(self) -> tuple:
        return self.reward_map.shape

    def actions(self) -> list:
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def is_wall(self, state: Tuple[int, int]) -> bool:
        if isinstance(self.wall_state, tuple):
            return state == self.wall_state
        elif isinstance(self.wall_state, list):
            return state in self.wall_state
        else:
            raise NotImplementedError


    def next_state(
        self,
        state: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        # 移動先の場所の計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]

        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 移動先がマップ外ではないかのチェック
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            # マップ外
            next_state = state

        # 移動先が壁ではないかのチェック
        if self.is_wall(next_state):
            next_state = state

        return next_state

    def reward(
        self,
        state: Tuple[int, int],
        action: int,
        next_state: Tuple[int, int]
    ) -> Optional[float]:
        return self.reward_map[next_state]

    def reset(self) -> Tuple[int, int]:
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v: Optional[dict] = None, policy=None, print_value: bool = True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value: bool = True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_q(q, print_value)


if __name__ == "__main__":
    env = GridWorld()
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
    env.render_v()
