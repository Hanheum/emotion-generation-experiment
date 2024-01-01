from emotion_v0 import emotion_v0
from PIL import Image
import numpy as np

env = emotion_v0(render_mode='human')

observation, info = env.reset()
env.helper_location = np.array([320, 320], dtype=np.float32)
env.enemies_captured = [True, True, True]
env.enemies_colors = np.array([env.blue, env.blue, env.blue])
observation = observation.astype(np.uint8)
#img = Image.fromarray(observation)
#img = img.resize((128, 128))
#mg.save('/home/hanheum/Desktop/emotion_render_test.png')


for i in range(3000):
    #forward
    action = 2

    observation, reward, done, info = env.step(action)
    if done:
        env.close()