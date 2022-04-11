from collections import deque
import gym
import random
import numpy as np
import torch

from TD3 import TD3

env_name = "MountainCarContinuous-v0" # multiple goals

# The following code is provided for the training of your agent in the 'BipedalWalker-v3' gym environment.
gym.logger.set_level(40)
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_upper_bound = 1  # action space upper bound
action_lower_bound = -1  # action space lower bound
# action_bound = env.action_space.high[0]
# action_clip_low = np.array([-1.0 * action_bound])
# action_clip_high = np.array([action_bound])


env.reset()
env.seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

timesteps_count = 0  # Counting the time steps
max_steps = 1600  # Maximum time steps for one episode
ep_reward_list = deque(maxlen=50)
avg_reward = -9999
agent = TD3(state_size, action_size, action_upper_bound, action_lower_bound)


# we run 600 episodes
for ep in range(10):
    state = env.reset()
    episodic_reward = 0
    timestep_for_cur_episode = 0

    for st in range(max_steps):
        # Select action according to policy
        action = agent.policy(state)
        action = np.array([action])

        # Recieve state and reward from environment.
        next_state, reward, done, info = env.step(action)
        episodic_reward += reward
        
        # Send the experience to the agent and train the agent
        agent.train(timesteps_count, timestep_for_cur_episode, state, action, reward, next_state, done)

        timestep_for_cur_episode += 1     
        timesteps_count += 1
        
        # env.render()
        # End this episode when `done` is True
        if done:
            break
        state = next_state

    ep_reward_list.append(episodic_reward)
    print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')
    
    if len(ep_reward_list) == 50:
        # Mean of last 50 episodes
        avg_reward = sum(ep_reward_list) / 50
        print(', Moving Average Reward: {:.2f}'.format(avg_reward))
    else:
        print('')

print('Average reward over 50 episodes: ', avg_reward)
env.close()

# Save the actor
actor_path = "actor.pth"
torch.save(agent.actor.state_dict(), actor_path)


