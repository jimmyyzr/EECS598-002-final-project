from random import seed
import gym
from gym.spaces import Discrete, Dict, Box
import os
import numpy as np
import torch
import torch.nn as nn
from EnvModel import EnvModel
from MPC import MPCController
from collections import deque

os.environ['LANG']='en_US'
max_steps = 100
env_name ="FetchReach-v1"
env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)
env_sim = EnvModel()
device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps_count = 0  # Counting the time steps

ep_reward_list = deque(maxlen=50)
# initial random sampling 
state_dic = env.reset()
state = state_dic['observation']
for _ in range(5000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  next_state_dic, reward, done, info = env.step(action)
  next_state = next_state_dic['observation']
  # store data into env model
  env_sim.store_data(state,action,reward,next_state)
  # store data into replaybuff
  state = next_state
  if done:
    observation = env.reset()

model = nn.Sequential(
  nn.Linear(14, 128),
  nn.ReLU(),
  nn.Linear(128, 256),
  nn.ReLU(),
  nn.Linear(256, 11)    
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.MSELoss()

for i in range(500):
  optimizer.zero_grad()
  train_x, train_y = env_sim.random_sampling()
  train_x = torch.tensor(train_x, device=device, dtype=torch.float)
  train_y = torch.tensor(train_y, device=device, dtype=torch.float)

  outputs = model(train_x)
  loss = loss_fn(outputs, train_y)
  loss.backward()
  optimizer.step()

  if i % 10 == 0:
    print(loss.item())




