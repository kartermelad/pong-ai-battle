from pong_env import PongEnv
from stable_baselines3 import PPO

env = PongEnv()
ppo_model = PPO.load("pong_agent")

obs, _ = env.reset()
done = False
score = 0
step_count = 0

try:
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        step_count += 1
        env.render()

    print(f"Game finished in {step_count} steps with score {score}")
finally:
    env.close()
