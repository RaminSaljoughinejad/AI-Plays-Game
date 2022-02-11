import os
import gym
from stable_baselines3 import PPO

model_name = "PPO"
env_name = "CartPole-v0"

model_dir = f"models/{model_name}"
logs_dir = f"logs/{model_name}"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = gym.make(env_name)
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS = 10000
for i in range(1,200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{model_dir}/{TIMESTEPS*i}")
env.close()