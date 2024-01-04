import tensorflow as tf
import numpy as np

class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super().__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.memory_vector_size = 128

        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(128, 128, 3))
        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate()
        self.m_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.m_layer2 = tf.keras.layers.Dense(self.memory_vector_size, activation='relu')
        self.e_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.e_layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.q_layer = tf.keras.layers.Dense(action_size)

    def __call__(self, x, memory):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.concat([x, memory])
        x = self.m_layer1(x)
        x = self.m_layer2(x)
        new_memory = x
        x = self.e_layer1(x)
        x = self.e_layer2(x)
        q_values = self.q_layer(x)
        return q_values, new_memory
    
    def reset_memory(self):
        self.memory_vector = np.zeros([1, self.memory_vector_size], dtype=np.float32)
    
