from random import seed
import gym
from gym.spaces import Discrete, Dict, Box
import os
import numpy as np
import torch
from EnvModel import EnvModel
from train_td3_reward_shaping import TD3
os.environ['LANG']='en_US'


agent = TD3()
actor = torch.load('actor.pth')
agent.actor.load_state_dict(actor)
max_steps = 100
env_name ="FetchReach-v1"
env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)
state_dic = env.reset()
desire_goal = state_dic["desired_goal"]
state = desire_goal - state_dic["observation"][0:3]


for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  action = agent.policy(state) # your agent here (this takes random actions)

  next_state_dic, reward, done, info = env.step(action)
  print(reward)
  env.render()

  next_state = desire_goal-next_state_dic["observation"][0:3]
  state = next_state
  if done or info["is_success"]:
    state_dic = env.reset()
    desire_goal = state_dic["desired_goal"]
    state = desire_goal - state_dic["observation"][0:3]
    print(info)
    
 
env.close()



