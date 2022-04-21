from stable_baselines3 import DDPG, TD3M
from stable_baselines3.common.noise import NormalActionNoise
import gym
import numpy as np

from Pbuffer import Pbuffer

from MPC import MPCController
from EnvModel import EnvModel


def virtual_sim(states, goals, env_sim):
    env_sim.update_model() # TODO: buggy, might because env_sim.store_data stores batch instead of single data

    mpc_actor.dyna_model = env_sim
    
    # number of virtual episodes
    num_episode = 200



    # sample initial state and goal from env.replay_buffer



    state_virtual = state
    for i in P : # TODO: Don't use for, vectorize it. 
        action = mpc_actor.get_best_action(state_virtual, goal)
        reward_vitrual, next_state_virtual = env_sim.step(state_virtual,action)
        state_virtual = next_state_virtual 


    # dones can be calculated by env

    return (observations,  # Dict[str, torch.tensor]
            next_observations, # Dict[str, torch.tensor]
            actions, 
            rewards,
            dones)

model_class = TD3M  # works also with SAC, DDPG and TD3
ENV_NAME = "FetchReach-v1"
MAX_EPISODE_STEPS = 100 # Time limit for the episodes
TIMESTEPS = int(2e3)


env = gym.make(ENV_NAME).unwrapped
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# You must use `MultiInputPolicy` when working with dict observation space, not MlpPolicy
# If replay_buffer_class not identified, or is HerReplayBuffer, n_envs is 1 by default
model = model_class('MultiInputPolicy', 
                    env, 
                    buffer_size = 1000, #TODO: now I use small size for fast testing
                    action_noise=action_noise, 
                    verbose=1)


P = Pbuffer()
state_dic = env.reset()

env_sim = EnvModel()
mpc_actor = MPCController(1, 5, env, env_sim)

last_buffer_pos = 0 # buffer pointer, used with current buffer pointer to index new transitions
for i in range(10):

    model.learn(total_timesteps=TIMESTEPS, log_interval=10)

    # model.replay_buffer.pos # buffer pointer
    # model.replay_buffer.size() size of contained data, == buffer_size once buffer is full
    
    # Get new transitions
    recent_data = model.replay_buffer.index(last_buffer_pos)
    
    state = recent_data.observations["observation"].cpu().detach().numpy() # shape (X, 10)
    goal = recent_data.observations["desired_goal"].cpu().detach().numpy()
    next_state = recent_data.next_observations["observation"].cpu().detach().numpy() 
    action = recent_data.actions.cpu().detach().numpy() 
    rewards = recent_data.rewards.cpu().detach().numpy() 

    print(rewards.shape)
    print(next_state.shape)
    exit(0)


    env_sim.store_data(state, action, rewards, next_state)# 10, 4, 1, 10
    v_experience = virtual_sim(state, goal, env_sim) # here I get state and goal from recent_data, I can also randomly sample the whole replay buffer

    # P.add(replay_data.observations, 
    #       replay_data.next_observations,
    #       replay_data.actions,
    #       replay_data.rewards,
    #       replay_data.dones)

    P.add(*v_experience) # TODO: check input is torch.tensor

    # model.train(gradient_steps=1, replay_buffer=P)



# model.save("model_path")