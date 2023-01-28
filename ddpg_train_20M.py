import gym
import tensorflow as tf
import os
import datetime
import stable_baselines3
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from stable_baselines3 import DDPG
from gym.envs.registration import registry, register, make, spec
from stable_baselines3.common.vec_env import SubprocVecEnv
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
  )
import os
models_dir = "models/DDPG"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

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
  envv = SubprocVecEnv([make_env for i in range(n_cpu)])

  config = tf.compat.v1.ConfigProto()

  with tf.compat.v1.Session(config=config):
  
    model = DDPG("MlpPolicy", env, learning_rate=0.0001,gamma=0.99, verbose=2,tensorboard_log=logdir)
    model.learn(total_timesteps=timestep, reset_num_timesteps=False, tb_log_name="DDPG")#15M timesteps and overnight run on a Macbook worked fine(still can improve).
  
    model.save(filename)
    model.save('./model2/'+filename)
  
    print('Model Saved')
