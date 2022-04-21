from random import seed
import gym
from gym.spaces import Discrete, Dict, Box
import os
import numpy as np
from EnvModel import EnvModel
from MPC2 import MPCController
from collections import deque
from train_td3_reward_shaping import TD3
import torch

os.environ['LANG']='en_US'
max_steps = 100
env_name ="FetchReach-v1"
env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)
env_sim = EnvModel()
mpc_actor = MPCController(2,100,env,env_sim)
TD3_actor = TD3()
timesteps_count = 0  # Counting the time steps

ep_reward_list_mpc = []
ep_reward_list_td3 = []

# initial random sampling 
# state_dic = env.reset(seed=10)
# state = state_dic['observation'][0:3]
# goal = state_dic["desired_goal"]
for _ in range(10):
  state_dic = env.reset()
  state = state_dic['observation'][0:3]
  desire_goal = state_dic["desired_goal"]
  for _ in range(50):
    action = env.action_space.sample() # your agent here (this takes random actions)
    next_state_dic, reward, done, info = env.step(action)
    next_state = next_state_dic['observation'][0:3]
    # a = np.sum(np.sqrt((desire_goal-next_state)**2))
    # reward  = a * reward
    # store data into env model
    env_sim.store_data(state,action,reward,next_state)
    # store data into replaybuff
    s = np.concatenate((state, desire_goal))
    s_next = np.concatenate((next_state, desire_goal))
    TD3_actor.add_to_replay_memory(s,action,reward,s_next,done)

    state = next_state
    if done:
      state_dic = env.reset()


for ep in range(600):
  # Initial reset for each episode
  state_dic = env.reset()
  desire_goal = state_dic["desired_goal"]
  state = state_dic["observation"][0:3]
        
  episodic_reward = 0
  timestep_for_cur_episode = 0
  # update world model each episode1
  env_sim.update_model()
  mpc_actor.dyna_model = env_sim

  for step in range(max_steps):
    env.render()
    action = mpc_actor.get_best_action(state,desire_goal)
    next_state_dic, reward, done, info = env.step(action)
    # a = np.sum(np.sqrt((desire_goal-next_state)**2))
    # reward  = a * reward
    next_state = next_state_dic["observation"][0:3]
    episodic_reward += reward

    # store data into env model
    env_sim.store_data(state,action,reward,next_state)
    # store data into replaybuff and train
    s = np.concatenate((state, desire_goal))
    s_next = np.concatenate((next_state, desire_goal))
    # TD3_actor.add_to_replay_memory(s,action,reward,s_next,done)
    TD3_actor.train(timesteps_count, timestep_for_cur_episode, s, action, reward, s_next, done)
    state = next_state
    timestep_for_cur_episode += 1     
    timesteps_count += 1

    # End this episode when `done` is True
    if done or info["is_success"]:
      state_dic = env.reset()
      desire_goal = state_dic["desired_goal"]
      state = state_dic["observation"][0:3]
      # break

  ep_reward_list_mpc.append(episodic_reward)
  print('Ep. {}, Ep.Timesteps {}, Episode Reward_MPC: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='\n',)
        
  
  # Initial reset for each episode2
  state_dic = env.reset()
  desire_goal = state_dic["desired_goal"]
  state = state_dic["observation"][0:3]
        
  episodic_reward = 0
  timestep_for_cur_episode = 0

  for step in range(max_steps):
    env.render()
    s = np.concatenate((state, desire_goal))
    action = TD3_actor.policy(s)
    next_state_dic, reward, done, info = env.step(action)
    next_state = next_state_dic["observation"][0:3]
    # a = np.sum(np.sqrt((desire_goal-next_state)**2))
    # reward  = a * reward
   
    
    episodic_reward += reward

    # store data into env model
    env_sim.store_data(state,action,reward,next_state)
    # store data into replaybuff and train
    s = np.concatenate((state, desire_goal))
    s_next = np.concatenate((next_state, desire_goal))
    # TD3_actor.add_to_replay_memory(s,action,reward,s_next,done)
    TD3_actor.train(timesteps_count, timestep_for_cur_episode, s, action, reward, s_next, done)
    state = next_state

    timestep_for_cur_episode += 1     
    timesteps_count += 1
    # End this episode when `done` is True
    if done or info["is_success"]:
      state_dic = env.reset()
      desire_goal = state_dic["desired_goal"]
      state = state_dic["observation"][0:3]
      
      break
   
  ep_reward_list_td3.append(episodic_reward)
  print('Ep. {}, Ep.Timesteps {}, Episode Reward_for_TD3: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='\n')
        



actor_path = "actor_TD3.pth"
torch.save(TD3_actor.actor.to("cpu").state_dict(), actor_path)
np.save('ep_reward_list_td3',ep_reward_list_td3)
np.save('ep_reward_list_mpc',ep_reward_list_mpc)







