import numpy as np
import gymnasium as gym
from math import sin, cos, radians
import random

import pygame
from gymnasium import spaces

class emotion_v0(gym.Env):
    metadata = {'render_modes':['human', 'rgb_array'], 'render_fps':24}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict(
            {
                'state': spaces.Box(0, 255, shape=(64, 64), dtype=int)
                #'state': spaces.Box(0, 64, shape=(informations about state), dtype=float)
            }
        )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)  #forward, backward, turn left, turn right, no operation

        self.window = None
        self.clock = None

        self.helper_location = None
        self.with_helper = False
        self.enemies = np.zeros((3, 2), dtype=np.float32)

        self.red = np.array((255, 0, 0))
        self.blue = np.array((0, 0, 255))
        self.green = np.array((0, 255, 0))
        self.white = np.array((255, 255, 255))
        self.black = np.array((0, 0, 0))

        self.enemies_colors = np.array([self.red, self.red, self.red])
        self.enemies_captured = [False, False, False]

        self.health = 100

        self.enemy_design = np.array([
            [-5, 5],
            [5, 5],
            [0, -10]
        ])

        self.enemies_rotated_designs = np.zeros([3, 3, 2], dtype=np.float32)
        self.enemies_moving_vector = [[0, -3], [0, -3], [0, -3]]
        self.enemies_moving_vector = np.array(self.enemies_moving_vector, dtype=np.float32)

        self.agent_design = np.array([
            [-7, 7],
            [7, 7],
            [0, -14]
        ])

        self.agent_rotation = 0

        self.foods = np.zeros([10, 2])

        self.agent_moving_vector = np.array([0, -5], dtype=np.float32)

    def get_info(self):
        info = {
            'health': self.health,
        }
        return info
    
    def rotate(self, vector, theta):
        theta = radians(theta)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        rotation_matrix = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])

        rotated_vector = np.matmul(vector, rotation_matrix)
        return rotated_vector
    
    def rotate_with_sin_cos(self, vector, sin_theta, cos_theta):
        rotation_matrix = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])

        rotated_vector = np.matmul(vector, rotation_matrix)
        return rotated_vector
    
    def render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((640, 640))
        
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((640, 640))
        canvas.fill(self.black)

        pygame.draw.line(canvas, self.white, (160, 480), (160, 640), width=5)
        pygame.draw.line(canvas, self.white, (480, 480), (480, 640), width=5)
        pygame.draw.line(canvas, self.white, (160, 480), (480, 480), width=5)
        
        #draw helper
        pygame.draw.circle(canvas, self.blue, self.helper_location.astype(int), 10)

        #draw enemies
        for i, enemy in enumerate(self.enemies):
            enemy_polygon_points = np.array([enemy, enemy, enemy]) + self.enemies_rotated_designs[i]
            enemy_polygon_points = np.round(enemy_polygon_points).astype(int)
            pygame.draw.polygon(canvas, self.enemies_colors[i], enemy_polygon_points, 5)

        #draw agent
        if self.agent_rotation > 360: 
            self.agent_rotation -= 360
        elif self.agent_rotation < -360:
            self.agent_rotation += 360
        rotated_agent_design = self.rotate(self.agent_design, self.agent_rotation)
        agent_polygon_points = np.array([self._agent_location, self._agent_location, self._agent_location]) + rotated_agent_design
        pygame.draw.polygon(canvas, (255, 255, 255), agent_polygon_points, 5)

        #draw foods
        for food in self.foods:
            pygame.draw.circle(canvas, (0, 255, 0), food, 3)

        #rendering for human
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.health = 100

        self.agent_moving_vector = np.array([0, -5], dtype=np.float32)
        self.agent_rotation = 0

        #self._agent_location = np.array([320, 600], dtype=np.float32)
        self._agent_location = np.random.random_integers(160, 480, size=2)
        self._agent_location = self._agent_location.astype(np.float32)
        self.helper_location = self.np_random.integers(160, 480, size=2, dtype=int)
        self.helper_location = self.helper_location.astype(np.float32)

        self.enemies_moving_vector = [[0, -3], [0, -3], [0, -3]]
        self.enemies_moving_vector = np.array(self.enemies_moving_vector, dtype=np.float32)
        for enemy in range(3):
            enemy_rotation = random.randint(0, 360)
            enemy_location = self.np_random.integers(160, 480, size=2, dtype=int)
            self.enemies[enemy] = enemy_location.astype(np.float32)
            self.enemies_rotated_designs[enemy] = self.rotate(self.enemy_design, enemy_rotation)
            self.enemies_moving_vector[enemy] = self.rotate(self.enemies_moving_vector[enemy], enemy_rotation)

        for food in range(10):
            self.foods[food] = self.np_random.integers(160, 480, size=2, dtype=int)

        observation = self.render_frame()
        info = self.get_info()

        return observation, info

    def turn_towards(self, design, A, B):
        #A: location of enemy
        #B: location of target(agent)

        sin_theta, cos_theta = self.normalize(B-A)
        cos_theta *= -1

        rotated_design = self.rotate_with_sin_cos(design, sin_theta, cos_theta)
        return rotated_design
    
    def distance2D(self, A, B):
        distance = A-B
        distance = distance**2
        distance = np.sum(distance)
        return distance**0.5
    
    def normalize(self, vector):
        size = self.distance2D(np.zeros([2]), vector)
        return vector/size

    def step(self, action):
        reward = 0
        done = False

        #agent action related
        if action == 0:
            #no operation
            pass
        elif action == 1:
            #backward
            self._agent_location -= self.agent_moving_vector
        elif action == 2:
            #forward
            self._agent_location += self.agent_moving_vector
        elif action == 3:
            #turn right
            self.agent_rotation += 5
            self.agent_moving_vector = self.rotate(self.agent_moving_vector, 5)
        elif action == 4:
            #turn left
            self.agent_rotation += -5
            self.agent_moving_vector = self.rotate(self.agent_moving_vector, -5)

        self._agent_location = np.clip(self._agent_location, 0, 639)

        #eat food
        for i, food in enumerate(self.foods):
            if self.distance2D(self._agent_location, food) <= 15:
                self.foods[i] = np.random.random_integers(160, 480, size=2)
                self.health += 1
                self.health = np.clip(self.health, 0, 100)
                reward += 1

        #helper logics
        if not self.with_helper and self.distance2D(self._agent_location, self.helper_location) <= 15:
            self.with_helper = True

        if self.with_helper:
            self.helper_location = self._agent_location.copy()
            self.helper_location[1] += 10
        self.helper_location = np.clip(self.helper_location, 0, 639)
        
        #enemy movements
        for i, enemy in enumerate(self.enemies):
            if self.distance2D(self._agent_location, enemy) <= 15:
                if not self.with_helper and not self.enemies_captured[i]:
                    self.health -= 5
                    reward -= 5
                else:
                    self.enemies_captured[i] = True
                    self.enemies_colors[i] = self.blue
                    self.health += 5
                    reward += 5

                    self.with_helper = False
                    self.helper_location = np.random.random_integers(160, 480, size=2)
                    self.helper_location = self.helper_location.astype(np.float32)

            vec_E2A = self.normalize(self._agent_location - enemy)
            if self.distance2D(enemy, self._agent_location) < 70 and vec_E2A.dot(self.normalize(self.enemies_moving_vector[i])) > 0.3 and self.enemies_captured[i]==False:
                self.enemies_rotated_designs[i] = self.turn_towards(self.enemy_design, enemy, self._agent_location)
                self.enemies_moving_vector[i] = vec_E2A*3
                self.enemies[i] += self.enemies_moving_vector[i]
            else:
                if self.enemies_captured[i]:
                    for f, food in enumerate(self.foods):
                        if self.distance2D(enemy, food) <= 15:
                            self.enemies_captured[i] = False
                            self.enemies_colors[i] = self.red
                            self.foods[f] = np.random.random_integers(160, 480, size=2)

                x, y = enemy

                is_in_safe_space = 480 <= y <= 640 and 160 <= x <= 480

                if x <= 0 or x >= 639 or y <= 0 or y >= 639 or is_in_safe_space:
                    theta = random.randint(90, 270)
                    self.enemies_rotated_designs[i] = self.rotate(self.enemies_rotated_designs[i], theta)
                    self.enemies_moving_vector[i] = self.rotate(self.enemies_moving_vector[i], theta)
                
                self.enemies[i] += self.enemies_moving_vector[i]

            self.enemies[i] = np.clip(self.enemies[i], 0, 639)

        info = self.get_info()
        observation = self.render_frame()

        if self.health <= 0:
            done = True
        
        return observation, reward, done, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()