import gym
import os
from stable_baselines3 import PPO


model_name = "PPO"
env_name = "CartPole-v0"
model_dir = f"models/{model_name}"
model_path = f"{model_dir}/620000.zip"
env = gym.make(env_name)
env.reset()


model = PPO.load(model_path, env=env)

n_run = 10
for i in range(n_run):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
env.close()

