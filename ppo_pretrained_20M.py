import gym
import stable_baselines3
from stable_baselines3 import DDPG
from gym.wrappers.record_video import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.envs.registration import registry, register, make, spec

#registering custom environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
)
if __name__ == '__main__':

# Create and wrap the environment
  env = gym.make('RocketLander-v0')
  env = RecordVideo(env, './video4-0',  episode_trigger = lambda episode_number: True)
  env.reset()



# Load the trained agent
#ppo2_RocketLander-v0_20000000_2019-05-05 03/26/38.pkl
  model = DDPG.load("/content/SMOPS-2023/model2/ppo2_RocketLander-v0_2000000_2023-01-23 08:04:12.zip")


#Trained agent in action
  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()
      if dones:
    	  break
