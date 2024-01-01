import tensorflow as tf
import numpy as np
from GridWorld import GridWorldEnv
import random
from collections import deque
from PIL import Image

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=state_size)
        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.q_layer = tf.keras.layers.Dense(action_size)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        q_values = self.q_layer(x)
        return q_values
    
class Agent:
    def __init__(self, action_size, state_size=(64, 64, 1)):
        self.action_size = action_size
        self.state_size = state_size

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=10.)

        self.memory = deque(maxlen=10000)

        self.epsilon = 0.02
        self.epsilon_end = 0.02
        self.epsilon_step = (1-0.02)/10000

        self.discount_rate = 0.99
        self.batch_size = 32
        self.start_train = 50000
        
        self.update_target_rate = 1000
        self.update_target()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def policy(self, x):
        if self.epsilon >= np.random.rand():
            return random.randrange(self.action_size)
        else:
            q_values = self.model(np.reshape(x, [1, 64, 64, 1]))
            return np.argmax(q_values[0])
        
    def train(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch], dtype=np.float32)

        with tf.GradientTape() as tape:
            prediction = self.model(states)
            prediction = prediction * tf.one_hot(actions, self.action_size)
            prediction = tf.reduce_sum(prediction, axis=1)

            max_q = self.target_model(next_states)
            max_q = np.amax(max_q, axis=1)

            error = tf.abs(rewards + (1-dones)*self.discount_rate*max_q - prediction)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return loss
    
env = GridWorldEnv(render_mode='rgb_array')
agent = Agent(4, (64, 64, 1))
agent.model.load_weights('./gridworld/gridworld')
agent.update_target()

episodes = 10
global_steps = 0
for episode in range(episodes):
    done = False

    obs, info = env.reset()
    state = env.render()
    state = state/255.

    sum_reward = 0
    steps = 0

    states = []
    while not done:
        global_steps += 1
        steps += 1
        action = agent.policy(state)
        obs, reward, done, _, info = env.step(action)
        sum_reward += reward
        next_state = env.render()/255.

        states.append(state)

        if steps > 300:
            done = True

        if done:
            states.append(next_state)
        
        state = next_state

    print('episode: {} | reward: {}'.format(episode+1, sum_reward))

    states = np.array(states, dtype=np.float32)
    states = states * 255.
    states = states.astype(np.uint8)
    states = np.reshape(states, [states.shape[0], 64, 64])
    
    frames = []
    for frame in states:
        frame = Image.fromarray(frame)
        frames.append(frame)

    frame_one = frames[0]
    frame_one.save('./result_gifs/episode{}.gif'.format(episode+1), format='GIF', append_images=frames, save_all=True, duration=300, loop=0)