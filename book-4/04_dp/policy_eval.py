from typing import Any, Dict, DefaultDict

from collections import defaultdict

from gridworld import GridWorld


def eval_onestep(
    pi: DefaultDict[Any, Dict[int, float]],
    V: DefaultDict[Any, float],
    env: GridWorld,
    gamma: float = 0.9
) -> DefaultDict[Any, float]:
    for state in env.states():  # 各状態へアクセス
        if state == env.goal_state:  # ゴールの価値関数は常に0
            V[state] = 0
            continue

        if env.is_wall(state):
            V[state] = 0
            continue

        action_probs = pi[state]  # probsはprobabilitiesの略
        new_V = 0

        # 各行動にアクセス
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 新しい価値関数
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(
    pi: DefaultDict[Any, Dict[int, float]],
    V: DefaultDict[Any, float],
    env: GridWorld,
    gamma: float,
    threshold: float = 0.001
) -> DefaultDict[Any, float]:
    while True:
        old_V = V.copy()  # 更新前の価値関数
        V = eval_onestep(pi, V, env, gamma)

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
    # ランダムな方策
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    env = GridWorld()
    gamma = 0.9

    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
