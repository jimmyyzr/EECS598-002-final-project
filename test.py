import gym
import time

max_steps = 100
env_name ="FetchReach-v1"
env = gym.make(env_name).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)


obs = env.reset(seed=10)
env.render()
for i in range(5):
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(info)      
        if done:
            obs = env.reset(seed=10)
			# print(info)
        # time.sleep(0.5)