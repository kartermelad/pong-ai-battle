import gymnasium as gym
import numpy as np
import pygame as pg
import random

class PongEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.screen_width = 800
        self.screen_height = 600
        self.paddle_width = 10
        self.paddle_height = 60
        self.paddle_speed = 8
        self.paddle1_y = self.screen_height // 2
        self.paddle2_y = self.screen_height // 2
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        self.ball_speed = 4
        self.ball_vel_x = self.ball_speed * random.choice([1, -1])
        self.ball_vel_y = self.ball_speed * random.choice([1, -1])
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.paddle_hits = 0
        self.steps = 0

        self.action_space = gym.spaces.Discrete(3)
        low = np.array([0, 0, -1, -1, 0, 0, -1, -1, -1, 0, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        pg.init()
        self.screen = None
        self.clock = pg.time.Clock()
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.font = pg.font.SysFont(None, 24)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        self.paddle1_y = self.screen_height // 2
        self.paddle2_y = self.screen_height // 2
        self.ball_speed = 4
        self.ball_vel_x = self.ball_speed * random.choice([1, -1])
        self.ball_vel_y = self.ball_speed * random.choice([1, -1])
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.paddle_hits = 0
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        current_speed = np.sqrt(self.ball_vel_x ** 2 + self.ball_vel_y ** 2)
        norm_ball_vel_x = self.ball_vel_x / current_speed if current_speed != 0 else 0
        norm_ball_vel_y = self.ball_vel_y / current_speed if current_speed != 0 else 0

        paddle_to_ball_y = (self.ball_y - self.paddle1_y) / (self.screen_height - self.paddle_height)
        speed_mag = np.log1p(current_speed) / np.log1p(100.0)
        velocity_angle = np.arctan2(self.ball_vel_y, self.ball_vel_x) / np.pi

        obs = np.array([
            self.ball_x / self.screen_width,
            self.ball_y / self.screen_height,
            norm_ball_vel_x,
            norm_ball_vel_y,
            self.paddle1_y / (self.screen_height - self.paddle_height),
            self.paddle2_y / (self.screen_height - self.paddle_height),
            self.paddle1_vel / self.paddle_speed,
            self.paddle2_vel / self.paddle_speed,
            paddle_to_ball_y,
            speed_mag,
            velocity_angle
        ], dtype=np.float32)
        return obs


    def step(self, action):
        self.steps += 1
        # truncated = self.steps >= 1000
        truncated = False
        terminated = False
        reward = -0.01

        # Paddle 1 control
        if action == 1:
            self.paddle1_vel = -self.paddle_speed
        elif action == 2:
            self.paddle1_vel = self.paddle_speed
        else:
            self.paddle1_vel = 0
        self.paddle1_y += self.paddle1_vel
        self.paddle1_y = max(0, min(self.paddle1_y, self.screen_height - self.paddle_height))

        # Paddle 2 auto-follow
        paddle2_center = self.paddle2_y + self.paddle_height / 2
        if self.ball_y < paddle2_center:
            self.paddle2_vel = -self.paddle_speed
        elif self.ball_y > paddle2_center:
            self.paddle2_vel = self.paddle_speed
        else:
            self.paddle2_vel = 0
        self.paddle2_y += self.paddle2_vel
        self.paddle2_y = max(0, min(self.paddle2_y, self.screen_height - self.paddle_height))

        # Ball movement
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        if self.ball_y <= 0 or self.ball_y >= self.screen_height:
            self.ball_vel_y *= -1
            self.ball_y = max(0, min(self.ball_y, self.screen_height))

        # Check paddle 1 collision
        if self.ball_x <= self.paddle_width:
            if self.paddle1_y <= self.ball_y <= self.paddle1_y + self.paddle_height:
                self.ball_x = self.paddle_width
                self.ball_vel_x *= -1
                self.paddle_hits += 1
                self.ball_speed += 0.5
                direction_x = 1
                direction_y = 1 if self.ball_vel_y > 0 else -1
                self.ball_vel_x = self.ball_speed * direction_x
                self.ball_vel_y = self.ball_speed * direction_y
                reward += 0.1  # hit reward
            else:
                reward = -1
                terminated = True
                return self._get_obs(), reward, terminated, truncated, {}

        # Check paddle 2 collision
        if self.ball_x >= self.screen_width - self.paddle_width:
            if self.paddle2_y <= self.ball_y <= self.paddle2_y + self.paddle_height:
                self.ball_x = self.screen_width - self.paddle_width
                self.ball_vel_x *= -1
                self.paddle_hits += 1
                self.ball_speed += 0.5
                direction_x = -1
                direction_y = 1 if self.ball_vel_y > 0 else -1
                self.ball_vel_x = self.ball_speed * direction_x
                self.ball_vel_y = self.ball_speed * direction_y
            else:
                reward = 1
                terminated = True
                return self._get_obs(), reward, terminated, truncated, {}

        # Distance penalty
        paddle_center = self.paddle1_y + self.paddle_height / 2
        distance_penalty = -abs(paddle_center - self.ball_y) / self.screen_height
        reward += 0.01 * distance_penalty

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.screen is None:
            self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
            pg.display.set_caption("Pong RL")

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return

        self.screen.fill(self.BLACK)

        paddle1_rect = pg.Rect(0, int(self.paddle1_y), self.paddle_width, self.paddle_height)
        paddle2_rect = pg.Rect(self.screen_width - self.paddle_width, int(self.paddle2_y), self.paddle_width, self.paddle_height)
        pg.draw.rect(self.screen, self.WHITE, paddle1_rect)
        pg.draw.rect(self.screen, self.WHITE, paddle2_rect)

        ball_rect = pg.Rect(int(self.ball_x), int(self.ball_y), self.paddle_width, self.paddle_width)
        pg.draw.ellipse(self.screen, self.GREEN, ball_rect)

        score_text = self.font.render(f"Hits: {self.paddle_hits}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))

        pg.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pg.quit()
            self.screen = None
