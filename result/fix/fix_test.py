from time import sleep
from stable_baselines3 import DDPG, TD3
import gym
import numpy as np

class ResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def reset(self):
        env.seed(0) # Choose the same seed as used in training 
        return self.env.reset()


model_class = DDPG  # works also with SAC, DDPG and TD3
ENV_NAME = "FetchReach-v1"
MAX_EPISODE_STEPS = 100 # Time limit for the episodes
TIMESTEPS = int(5e4)
model_path = "fix_ddpg"

env = gym.make(ENV_NAME).unwrapped
env = ResetWrapper(env) 
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


model = model_class.load(model_path, env=env)

# obs = env.reset()
# env.render()
for i in range(10):
	done = False
	obs = env.reset()
	while not done:
	# for i in range(10):
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			print(info)
			break
