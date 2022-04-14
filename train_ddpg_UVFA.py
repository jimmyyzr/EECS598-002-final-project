import gym
import numpy as np
import torch
from DDPG_UVFA import DDPG_UVFA
from collections import deque
from rbuffer import ReplayBuffer_GoalEnv

# parameters:
MAX_STEP = 1600            # Maximum time steps for one episode
update_every = 200
exploration_action_noise = 0.1

# DDPG parameters:
GAMMA = 0.95                # discount factor for future rewards
N_ITER = 100                # update policy n_iter times in one DDPG update
BATCH_SIZE = 100            # num of transitions sampled from replay buffer
LR = 0.001


# FetchReach-v1
# FetchPush-v1
# env_name = "MountainCarContinuous-v0"
env_name ="FetchReach-v1"
# env_name = "Pendulum-v1"

env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEP+1)


print("observation space:", env.observation_space['observation'])
print("action space:", env.action_space.shape)


state_size = env.observation_space['observation'].shape[0]
action_size = env.action_space.shape[0]
action_bound = env.action_space.high[0]
action_clip_low = np.array([-1.0 * action_bound])
action_clip_high = np.array([action_bound])


ddpg = DDPG_UVFA(state_size, action_size, action_bound, lr=LR, gamma=GAMMA)
replay_buffer = ReplayBuffer_GoalEnv()

state_dic = env.reset()
state = state_dic["observation"]

# First, play with random actions to fill Replay Buffer
for i in range(BATCH_SIZE * 6):
    action = env.action_space.sample()
    action = action + np.random.normal(0, exploration_action_noise)
    # action = action.clip(self.action_clip_low, self.action_clip_high)

    next_state_dic, reward, done, info = env.step(action)
    next_state = next_state_dic["observation"]
    desire_goal = next_state_dic["desired_goal"]
    replay_buffer.add((state, action, reward, next_state, desire_goal, float(done)))
    # env.render()

print("Start playing!")

ep_reward_list = deque(maxlen=50)
avg_reward = -9999
timesteps_count = 0  # Counting the time steps


# we run 600 episodes
for ep in range(600):
    state_dic = env.reset()
    state = state_dic["observation"]
    desire_goal = state_dic["desired_goal"]
    cur_time_step = 0
    episodic_reward = 0
    timestep_for_cur_episode = 0

    for st in range(MAX_STEP):
        # Select action according to policy
        action = ddpg.select_action(state,desire_goal)
        action = action.clip(action_clip_low, action_clip_high)
        
        # Recieve state and reward from environment.
        next_state_dic, reward, done, info = env.step(action)
        next_state = next_state_dic["observation"]
        desire_goal = next_state_dic["desired_goal"]
        # print("next_state_desire goal",next_state_dic["desired_goal"])
        replay_buffer.add((state, action, reward, next_state, desire_goal,float(done)))
        episodic_reward += reward
        # env.render()

        if cur_time_step % update_every == 0: # We update once every xx environment interations
            ddpg.update(replay_buffer, N_ITER, BATCH_SIZE)

        timestep_for_cur_episode += 1     
        timesteps_count += 1

        # End this episode when `done` is True
        if done:
            break
        state = next_state
        cur_time_step += 1

    
    ep_reward_list.append(episodic_reward)
    print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')
    
    if len(ep_reward_list) == 50:
        # Mean of last 50 episodes
        avg_reward = sum(ep_reward_list) / 50
        print(', Moving Average Reward: {:.2f}'.format(avg_reward))
    else:
        print('')

actor_path = "actor.pth"
torch.save(ddpg.actor.to("cpu").state_dict(), actor_path)
env.close()