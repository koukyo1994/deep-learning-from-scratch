import gym


env = gym.make("ALE/Pong-v5", render_mode="human", new_step_api=True)
for i in range(1):
    observation = env.reset()
    for t in range(1000):
        observation, reward, done, _, info = env.step(env.action_space.sample())
env.close()
