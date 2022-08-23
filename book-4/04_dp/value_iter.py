import argparse
from collections import defaultdict
from typing import DefaultDict

import numpy as np

from gridworld import GridWorld
from policy_iter import greedy_policy, State


def value_iter_onestep(
    V: DefaultDict[State, float],
    env: GridWorld,
    gamma: float
) -> DefaultDict[State, float]:
    for state in env.states():  # 全ての状態について評価と改善を1ステップずつ行う
        if state == env.goal_state:  # ゴールの価値関数は常に0
            V[state] = 0.0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            if r is None:
                r = 0.0
            value = r + gamma * V[next_state]  # 新しい価値関数
            action_values.append(value)

        V[state] = max(action_values)  # 最大値を取り出す
    return V


def value_iter(
    V: DefaultDict[State, float],
    env: GridWorld,
    gamma: float,
    threshold: float = 0.001,
    is_render: bool = True
) -> DefaultDict[State, float]:
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()  # 更新前の価値関数
        V = value_iter_onestep(V, env, gamma)

        # 更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        # 閾値との比較
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_map", action="store_true")
    args = parser.parse_args()

    V = defaultdict(lambda: 0)
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

    gamma = 0.9
    V = value_iter(V, env, gamma)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
