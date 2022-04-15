import gym
import numpy as np
from DDPG import DDPG
from collections import deque
from rbuffer import ReplayBuffer

# parameters:
max_steps = 1600            # Maximum time steps for one episode
update_every = 200
exploration_action_noise = 0.1

# DDPG parameters:
gamma = 0.95                # discount factor for future rewards
n_iter = 100                # update policy n_iter times in one DDPG update
batch_size = 100            # num of transitions sampled from replay buffer
lr = 0.001


# FetchReach-v1
# FetchPush-v1
# env_name = "MountainCarContinuous-v0"
env_name = "BipedalWalker-v3"
# env_name = "Pendulum-v1"

env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)


print("observation space:", env.observation_space.shape)
print("action space:", env.action_space.shape)

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_bound = env.action_space.high[0]
action_clip_low = np.array([-1.0 * action_bound])
action_clip_high = np.array([action_bound])

ddpg = DDPG(state_size, action_size, action_bound, lr, gamma)
replay_buffer = ReplayBuffer()

state = env.reset()
# First, play with random actions to fill Replay Buffer
for i in range(batch_size * 6):
    action = env.action_space.sample()
    action = action + np.random.normal(0, exploration_action_noise)
    action = action.clip(action_clip_low, action_clip_high)

    next_state, reward, done, info = env.step(action)
    replay_buffer.add((state, action, reward, next_state, float(done)))
    # env.render()

print("Start playing!")

ep_reward_list = deque(maxlen=50)
avg_reward = -9999
timesteps_count = 0  # Counting the time steps


# we run 600 episodes
for ep in range(600):
    state = env.reset()
    cur_time_step = 0
    episodic_reward = 0
    timestep_for_cur_episode = 0

    for st in range(max_steps):
        # Select action according to policy
        action = ddpg.select_action(state)
        action = action + np.random.normal(0, exploration_action_noise)
        action = action.clip(action_clip_low, action_clip_high)
        
        # Recieve state and reward from environment.
        next_state, reward, done, info = env.step(action)
        replay_buffer.add((state, action, reward, next_state, float(done)))
        episodic_reward += reward
        # env.render()

        if cur_time_step % update_every == 0: # We update once every xx environment interations
            ddpg.update(replay_buffer, n_iter, batch_size)

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

env.close()