import gym
import random
import numpy as np

class PongEnv(gym.Env):
    def __init__(self):
        self.screen_width = 400
        self.screen_height = 300
        self.paddle_width = 10
        self.paddle_height = 60
        self.paddle1_y = self.screen_height / 2
        self.paddle2_y = self.screen_height / 2
        self.ball_x = 200
        self.ball_y = 200
        self.ball_speed = 4
        self.ball_vel_x = self.ball_speed * random.choice([1, -1])
        self.ball_vel_y = self.ball_speed * random.choice([1, -1])
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.paddle_hits = 0
        self.action_space = gym.spaces.Discrete(3)
        low = np.array([0, 0, -1, -1, 0, 0, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.ball_x = 200
        self.ball_y = 150
        self.paddle1_y = self.screen_height / 2
        self.paddle2_y = self.screen_height / 2
        self.ball_speed = 4
        self.ball_vel_x = self.ball_speed * random.choice([1, -1])
        self.ball_vel_y = self.ball_speed * random.choice([1, -1])
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.paddle_hits = 0
        return self._get_obs()

    def _get_obs(self):
        paddle_speed = 8

        current_speed = np.sqrt(self.ball_vel_x ** 2 + self.ball_vel_y ** 2)
        if current_speed == 0:
            norm_ball_vel_x = 0
            norm_ball_vel_y = 0
        else:
            norm_ball_vel_x = self.ball_vel_x / current_speed
            norm_ball_vel_y = self.ball_vel_y / current_speed

        obs = np.array([
            self.ball_x / self.screen_width,
            self.ball_y / self.screen_height,
            norm_ball_vel_x,
            norm_ball_vel_y,
            self.paddle1_y / (self.screen_height - self.paddle_height),
            self.paddle2_y / (self.screen_height - self.paddle_height),
            self.paddle1_vel / paddle_speed,
            self.paddle2_vel / paddle_speed
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        paddle_speed = 8

        if action == 1:
            self.paddle1_vel = -paddle_speed
        elif action == 2:
            self.paddle1_vel = paddle_speed
        else:
            self.paddle1_vel = 0

        self.paddle1_y += self.paddle1_vel
        self.paddle1_y = max(0, min(self.paddle1_y, self.screen_height - self.paddle_height))
        obs = self._get_obs()
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info