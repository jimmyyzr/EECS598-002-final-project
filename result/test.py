from stable_baselines3 import DDPG, TD3M
from stable_baselines3.common.noise import NormalActionNoise
import gym
import numpy as np

from Pbuffer import Pbuffer

class ResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def reset(self):
        env.seed(0) # seed=0: below if initial state; seed=40 farther and more difficult
        return self.env.reset()


model_class = TD3M  # works also with SAC, DDPG and TD3
ENV_NAME = "FetchReach-v1"
MAX_EPISODE_STEPS = 100 # Time limit for the episodes
TIMESTEPS = int(5e3)
model_path = "ddpg_fixed.pth"


env = gym.make(ENV_NAME).unwrapped
env = ResetWrapper(env) 
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# You must use `MultiInputPolicy` when working with dict observation space, not MlpPolicy
# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model = model_class('MultiInputPolicy', env, action_noise=action_noise, verbose=1)


P = Pbuffer()

for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, log_interval=10)

    replay_data = model.replay_buffer.sample(batch_size=2)

    P.add(replay_data.observations, 
        replay_data.next_observations,
        replay_data.actions,
        replay_data.rewards,
        replay_data.dones)

    model.train(gradient_steps=1, replay_buffer=P)

    # model.learn(total_timesteps=TIMESTEPS, log_interval=10)


# model.save("model_path")