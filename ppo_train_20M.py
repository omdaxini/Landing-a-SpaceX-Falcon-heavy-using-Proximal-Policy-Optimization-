import gym
import tensorflow as tf
import os
import datetime
import stable_baselines3
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from stable_baselines3 import PPO
from gym.envs.registration import registry, register, make, spec
from stable_baselines3.common.vec_env import SubprocVecEnv
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
  )
if __name__ == '__main__':
  tensorboard = TensorBoard(log_dir='logs')
  #Register the RockerLander environment
  

  timestep = 2000000
  ENV  = 'RocketLander-v0'
  timestamp = datetime.datetime.now()
  filename = "ppo2_{}_{}_{}".format(ENV,timestep,str(timestamp)[:19])
  path = '{}_tensorboard'.format(ENV[:-3])
  env = gym.make('RocketLander-v0')
  def make_env():
    return gym.make('RocketLander-v0')
  n_cpu = 8
  env = SubprocVecEnv([make_env for i in range(n_cpu)])

  config = tf.compat.v1.ConfigProto()

  with tf.compat.v1.Session(config=config):
  
    model = PPO("MlpPolicy", env, learning_rate=0.0001,n_steps=256,gamma=0.99, verbose=2)
    model.learn(total_timesteps=timestep, log_interval=1000)#15M timesteps and overnight run on a Macbook worked fine(still can improve).
  
    model.save(filename)
    model.save('./model2/'+filename)
  
    print('Model Saved')
