from GridWorld import GridWorldEnv

env = GridWorldEnv(render_mode='human')

observation, info = env.reset()

for i in range(100):
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    print(reward, done)
    if done:
        observation, info = env.reset()