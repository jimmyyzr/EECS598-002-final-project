from random import seed
import gym
from gym.spaces import Discrete, Dict, Box
import os
import numpy as np
from EnvModel import EnvModel
from MPC import MPCController
from collections import deque

os.environ['LANG']='en_US'
max_steps = 100
env_name ="FetchReach-v1"
env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)
env_sim = EnvModel()
mpc_actor = MPCController(1,5,env,env_sim)

timesteps_count = 0  # Counting the time steps

ep_reward_list = deque(maxlen=50)
# initial random sampling 
state_dic = env.reset(seed=10)
state = state_dic['observation']
for _ in range(500):
  action = env.action_space.sample() # your agent here (this takes random actions)
  next_state_dic, reward, done, info = env.step(action)
  next_state = next_state_dic['observation']
  # store data into env model
  env_sim.store_data(state,action,reward,next_state)
  # store data into replaybuff
  state = next_state
  if done:
    observation = env.reset()


for ep in range(600):
  # Initial reset for each episode
  state_dic = env.reset()
  desire_goal = state_dic["desired_goal"]
  state = state_dic["observation"]
        
  episodic_reward = 0
  timestep_for_cur_episode = 0
  # update world model each episode
  env_sim.update_model()
  mpc_actor.dyna_model = env_sim

  for step in range(max_steps):
    env.render()
    action = mpc_actor.get_best_action(state,desire_goal)
    next_state_dic, reward, done, info = env.step(action)
    next_state = next_state_dic["observation"]
    episodic_reward += reward

    # store data into env model
    env_sim.store_data(state,action,reward,next_state)
    # store data into replaybuff 
    
    state = next_state

    timestep_for_cur_episode += 1     
    timesteps_count += 1
    # End this episode when `done` is True
    if done:
        break
   
  ep_reward_list.append(episodic_reward)
  print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')
        
  if len(ep_reward_list) == 50:
    # Mean of last 50 episodes
    avg_reward = sum(ep_reward_list) / 50
    print(', Moving Average Reward: {:.2f}'.format(avg_reward))
  else:
    print('')

print('Average reward over 50 episodes: ', avg_reward)
    



# for _ in range(1000):
#   # env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   next_state, reward, done, info = env.step(action)
#   env_sim.store_data(state['observation'],action,reward,next_state['observation'])
#   if _ > 10:
#     env_sim.update_model()
#     r,s_next = env_sim.generate_data(state['observation'],action)
#   state = next_state
#   if done:
#     observation = env.reset(seed=10)
#     print(observation["desired_goal"])
# env.close()



