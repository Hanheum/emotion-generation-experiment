import tensorflow as tf
import numpy as np
from emotion_v0_DQN import DQN
from collections import deque
import random

class Agent:
    def __init__(self, action_size=5, state_size=[128, 128, 3]):
        self.action_size = action_size
        self.state_size = state_size

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = tf.keras.optimizers.Adam()

        self.memory = deque(maxlen=100000)
        
        self.epsilon = 1.
        self.epsilon_end = 0.02
        self.epsilon_step = (1-0.02)/1000000

        self.discount_rate = 0.8

        self.batch_size = 32

        self.target_update_rate = 30000
        self.target_update()

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, x, memory):
        #shape of x will be [n, 128, 128, 3]
        #shape of memory will be [n, memory_size]
        if self.epsilon >= np.random.rand():
            return random.randrange(self.action_size)
        else:
            q_values = self.model(x, memory)
            return q_values
        
    def add_memory(self, state, action, reward, done, next_state, memory, next_memory):
        self.memory.append((state, action, reward, done, next_state, memory, next_memory))
    
    def train(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_step

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        dones = np.array([sample[3] for sample in batch])
        next_states = np.array([sample[4] for sample in batch], dtype=np.float32)
        memories = np.array([sample[5] for sample in batch], dtype=np.float32)
        next_memories = np.array([sample[6] for sample in batch], dtype=np.float32)

        with tf.GradientTape() as tape:
            q_values = self.model(states, memories)
            target_q = self.target_model(next_states, next_memories)

            q_values = q_values * tf.one_hot(actions, self.actino_size)
            q_values = tf.reduce_sum(q_values, axis=1)

            target_q = np.amax(target_q, axis=1)

            loss = (rewards + target_q*self.discount_rate*(1-dones) - q_values)**2
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))