import gym
import random
import numpy as np

class PongEnv(gym.Env):
    
    def __init__(self):
        self.screen_width = 400
        self.screen_height = 300
        self.paddle_width = 10
        self.paddle_height = 60
        self.paddle_speed = 8
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
            self.paddle1_vel / self.paddle_speed,
            self.paddle2_vel / self.paddle_speed
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        if action == 1:
            self.paddle1_vel = -self.paddle_speed
        elif action == 2:
            self.paddle1_vel = self.paddle_speed
        else:
            self.paddle1_vel = 0
        self.paddle1_y += self.paddle1_vel
        self.paddle1_y = max(0, min(self.paddle1_y, self.screen_height - self.paddle_height))

        paddle2_center = self.paddle2_y + self.paddle_height / 2
        if self.ball_y < paddle2_center:
            self.paddle2_vel = -self.paddle_speed
        elif self.ball_y > paddle2_center:
            self.paddle2_vel = self.paddle_speed
        else:
            self.paddle2_vel = 0
        self.paddle2_y += self.paddle2_vel
        self.paddle2_y = max(0, min(self.paddle2_y, self.screen_height - self.paddle_height))

        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vel_y *= -1
        elif self.ball_y >= self.screen_height:
            self.ball_y = self.screen_height
            self.ball_vel_y *= -1

        if self.ball_x <= self.paddle_width:
            if self.paddle1_y <= self.ball_y <= self.paddle1_y + self.paddle_height:
                self.ball_x = self.paddle_width 
                self.ball_vel_x *= -1
                self.paddle_hits += 1
                self.ball_speed += 0.5
                direction_x = 1 if self.ball_vel_x > 0 else -1
                direction_y = 1 if self.ball_vel_y > 0 else -1
                self.ball_vel_x = self.ball_speed * direction_x
                self.ball_vel_y = self.ball_speed * direction_y
            else:
                reward = -1
                done = True
                return self._get_obs(), reward, done, {}

        if self.ball_x >= self.screen_width - self.paddle_width:
            if self.paddle2_y <= self.ball_y <= self.paddle2_y + self.paddle_height:
                self.ball_x = self.screen_width - self.paddle_width
                self.ball_vel_x *= -1
                self.paddle_hits += 1
                self.ball_speed += 0.5
                direction_x = 1 if self.ball_vel_x > 0 else -1
                direction_y = 1 if self.ball_vel_y > 0 else -1
                self.ball_vel_x = self.ball_speed * direction_x
                self.ball_vel_y = self.ball_speed * direction_y
            else:
                reward = 1
                done = True
                return self._get_obs(), reward, done, {}

        reward = 0
        done = False
        return self._get_obs(), reward, done, {}