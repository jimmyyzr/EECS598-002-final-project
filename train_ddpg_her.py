import gym
import numpy as np
import torch
from DDPG import DDPG
from collections import deque
from herbuffer import HerReplayBuffer

# parameters:
max_steps = 100            # Maximum time steps for one episode
update_every = 200
exploration_action_noise = 0.1

# DDPG parameters:
gamma = 0.95                # discount factor for future rewards
n_iter = 100                # update policy n_iter times in one DDPG update
batch_size = 100            # num of transitions sampled from replay buffer
lr = 0.001

# HER parameters:
buffer_size = int(1e5)      # number of transitions stored in the buffer
rollout_steps = batch_size * 5 #* max_steps
assert(rollout_steps > max_steps) # self.n_episodes_stored must >=1 when sample transitions

# this HER can only run GoalEnv 
env_name = "FetchReach-v1"

env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

obs_dim = env.observation_space["observation"].shape[0]
goal_dim = env.observation_space["achieved_goal"].shape[0]

state_size = obs_dim + goal_dim
action_size = env.action_space.shape[0]
action_bound = env.action_space.high #[1. 1. 1. 1.]
action_clip_low = env.action_space.low #[-1. -1. -1. -1.]
action_clip_high = env.action_space.high #[1. 1. 1. 1.]

print("state_size", state_size)
ddpg = DDPG(state_size, action_size, action_bound, lr, gamma)
replay_buffer = HerReplayBuffer(env, buffer_size, max_episode_length=max_steps)


ep_reward_list = deque(maxlen=50)
avg_reward = -9999
timesteps_count = 0  # Counting the time steps

# rollout to fill experience buffer
env.reset()
# First, play with random actions to fill Replay Buffer
notdone = 0
for i in range(rollout_steps):
    action = env.action_space.sample()
    action = action + np.random.normal(0, exploration_action_noise)
    action = action.clip(action_clip_low, action_clip_high)
    obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, reward, done, info)
    # env.render()    
    if done:
        env.reset()

print("Start playing!")

# we run 600 episodes
for ep in range(600):
    obs = env.reset()
    state = np.concatenate((obs["observation"], obs["desired_goal"]))

    cur_time_step = 0
    episodic_reward = 0
    timestep_for_cur_episode = 0

    for st in range(max_steps):
        # Select action according to policy
        action = ddpg.select_action(state)
        action = action + np.random.normal(0, exploration_action_noise)
        action = action.clip(action_clip_low, action_clip_high)
        
        # Recieve state and reward from environment.
        obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, reward, done, info)
        episodic_reward += reward
        env.render()

        if cur_time_step % update_every == 0: # We update once every xx environment interations
            ddpg.update(replay_buffer, n_iter, batch_size)

        timestep_for_cur_episode += 1     
        timesteps_count += 1

        # End this episode when `done` is True
        if done:
            break

        state = np.concatenate((obs["observation"], obs["desired_goal"]))
        cur_time_step += 1

    ep_reward_list.append(episodic_reward)
    print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')
    
    if len(ep_reward_list) == 50:
        # Mean of last 50 episodes
        avg_reward = sum(ep_reward_list) / 50
        print(', Moving Average Reward: {:.2f}'.format(avg_reward))
    else:
        print('')

env.close()