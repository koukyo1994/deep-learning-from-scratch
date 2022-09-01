import numpy as np
import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v0", render_mode="human", new_step_api=True)
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = np.random.choice([0, 1])
        _, reward, done, _, _ = env.step(action)

    env.close()
