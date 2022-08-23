import argparse
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple

import numpy as np

from gridworld import GridWorld
from policy_eval import policy_eval

State = Tuple[int, int]
ActionProb = Dict[int, float]


def argmax(d: Dict[int, float]) -> int:
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(
    V: DefaultDict[State, float],
    env: GridWorld,
    gamma: float
) -> Dict[State, ActionProb]:
    """
    状態遷移関数が決定論的に振る舞うとして
    μ'(s) = argmax_a {r(s, a, s') + gamma v(s')}
    という式に従い新しい方策を得る
    """
    pi: Dict[State, ActionProb] = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            if r is None:
                r = 0.0
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {key: 0 for key in env.actions()}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(
    env: GridWorld,
    gamma: float,
    threshold: float = 0.001,
    is_render: bool = False,
) -> Dict[State, ActionProb]:
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)  # 評価
        new_pi = greedy_policy(V, env, gamma)  # 改善

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:  # 更新されていなかったら最適方策
            break
        pi = new_pi

    return pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_map", action="store_true")
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
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
