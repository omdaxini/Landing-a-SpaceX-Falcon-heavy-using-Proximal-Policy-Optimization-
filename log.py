import gym
import numpy as np
from gym.envs.registration import registry, register, make, spec
import os

from stable_baselines3 import A2C
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
  )
models_dir = "models/A2C3"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make("RocketLander-v0")
env.reset()

model = A2C('MlpPolicy', env, learning_rate= 0.0001,n_steps=256,gamma=0.99,verbose=2, tensorboard_log=logdir)
TIMESTEPS = 100000 
iters = 0
for i in range(20):
  model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C3")
  model.save(f"{models_dir}/{TIMESTEPS*i}")

#____next_cell_______
%load_ext tensorboard
%tensorboard --logdir /content/SMOPS-2023/logs/A2C3_0
