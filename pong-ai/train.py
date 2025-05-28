from pong_env import PongEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

env = DummyVecEnv([lambda: PongEnv(render_mode=None)])
model_path = "models/pong_agent.zip"

if os.path.exists(model_path):
    ppo_model = PPO.load("models/pong_agent", env=env)
    print("Loaded existing model")
else:
    ppo_model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/")
    print("Created new model")

ppo_model.learn(total_timesteps=100000, tb_log_name="PPO_run", reset_num_timesteps=True)
os.makedirs("models", exist_ok=True)
ppo_model.save("models/pong_agent")