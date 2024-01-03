from emotion_v0 import emotion_v0
from collections import deque
import numpy as np

env = emotion_v0()

observation, info = env.reset()
health = info['health']
done = False

memory = deque(maxlen=1000)

while not done:
    action = env.action_space.sample()
    next_observation, reward, done, info = env.step(action)
    health = info['health']

    memory.append((observation, action, reward, done, next_observation))
    
    observation = next_observation

    print(len(memory))

observations = [sample[0] for sample in memory]
observations = np.array(observations, dtype=np.float32)/255.

rewards = [sample[2] for sample in memory]

print(observations.shape)
print(rewards)