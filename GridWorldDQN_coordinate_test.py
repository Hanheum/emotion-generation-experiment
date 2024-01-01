import tensorflow as tf
import numpy as np
from GridWorld import GridWorldEnv
import random
from collections import deque

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = tf.keras.layers.Dense(10, activation='relu', input_shape=state_size)
        self.fc2 = tf.keras.layers.Dense(10, activation='relu')
        self.q_layer = tf.keras.layers.Dense(action_size)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q_values = self.q_layer(x)
        return q_values
    
class Agent:
    def __init__(self, action_size, state_size=(4, )):
        self.action_size = action_size
        self.state_size = state_size

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)

        self.optimizer = tf.keras.optimizers.Adam()

        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        
        self.update_target_rate = 10000
        self.start_train = 50000

        self.epsilon = 0.02
        self.epsilon_end = 0.02
        self.epsilon_step = (1-0.02)/100000

        self.discount_rate = 0.99

        self.update_target()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def policy(self, x):
        if self.epsilon >= np.random.rand():
            return random.randrange(self.action_size)
        
        else:
            q_values = self.model(np.reshape(x, [1, 4]))
            return np.argmax(q_values[0])

    def train(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        with tf.GradientTape() as tape:
            prediction = self.model(states)
            actions = tf.one_hot(actions, self.action_size)
            prediction = prediction * actions
            prediction = tf.reduce_sum(prediction, axis=1)

            max_q = self.target_model(next_states)
            max_q = np.amax(max_q, axis=1)
            
            loss = (rewards + (1-dones)*self.discount_rate*max_q - prediction)**2
            loss = tf.reduce_mean(loss)

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

env = GridWorldEnv(render_mode='human')
agent = Agent(4, (4, ))
agent.model.load_weights('./gridworld_coordinates/gridworld')

episodes = 10
global_steps = 0
steps_list = []
for episode in range(episodes):
    done = False
    steps = 0
    sum_reward = 0

    state, info = env.reset()

    while not done:
        global_steps += 1
        steps += 1
        action = agent.policy(state)
        next_state, reward, done, _, info = env.step(action)

        sum_reward += reward

        if steps >= 50:
            done = True

        state = next_state

    print('episode: {} | reward: {} | steps: {}'.format(episode, reward, steps))