import numpy as np

from gridworld import GridWorld


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

    V = {}
    for state in env.states():
        V[state] = np.random.randn()  # ダミーの状態価値関数
    env.render_v(V)
