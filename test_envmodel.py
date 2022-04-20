import gym
from gym.spaces import Discrete, Dict, Box
import os
import numpy as np
from EnvModel import EnvModel
os.environ['LANG']='en_US'

env = gym.make("FetchReach-v1")
env_sim = EnvModel()
state = env.reset(seed=10)
a = env.observation_space["desired_goal"]
print(state["desired_goal"])
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  next_state, reward, done, info = env.step(action)
  env_sim.store_data(state['observation'],action,reward,next_state['observation'])
  if _ > 10:
    env_sim.update_model()
    r,s_next = env_sim.step(state['observation'],action)
  state = next_state
  if done:
    observation = env.reset(seed=10)
    print(observation["desired_goal"])
env.close()



