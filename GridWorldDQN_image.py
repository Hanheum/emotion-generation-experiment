import tensorflow as tf
import numpy as np
from GridWorld import GridWorldEnv
import random
from collections import deque

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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=10.)

        self.memory = deque(maxlen=100000)

        self.epsilon = 1.
        self.epsilon_end = 0.02
        self.epsilon_step = (1-0.02)/100000

        self.discount_rate = 0.5
        self.batch_size = 32
        self.start_train = 30000
        
        self.update_target_rate = 10000
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

            error = tf.square(rewards + (1-dones)*self.discount_rate*max_q - prediction)
            loss = tf.reduce_mean(error)

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return loss
    
env = GridWorldEnv(render_mode='rgb_array')
agent = Agent(4, (64, 64, 1))
#agent.model.load_weights('./gridworld_img/gridworld_img')
#agent.update_target()

episodes = 2000
global_steps = 0
for episode in range(episodes):
    done = False

    obs, info = env.reset()
    state = env.render()
    state = state/255.

    sum_reward = 0
    steps = 0
    loss = 0
    while not done:
        global_steps += 1
        steps += 1
        action = agent.policy(state)
        obs, reward, done, _, info = env.step(action)
        sum_reward += reward
        next_state = env.render()/255.

        agent.add_memory(state, action, reward, next_state, done)
        
        if len(agent.memory) >= agent.start_train:
            loss += agent.train()
            if global_steps % agent.update_target_rate == 0:
                print('updating target model...')
                agent.update_target()

        if steps > 300:
            done = True
        
        state = next_state

    print('episode: {} | reward: {} | loss: {} | epsilon: {} | memory length: {}'.format(episode+1, sum_reward, loss/steps, agent.epsilon, len(agent.memory)))
    if (episode+1) % 100 == 0:
        agent.model.save_weights('./gridworld/gridworld')