from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import time
import gym

model_class = TD3  # works also with SAC, DDPG and TD3
ENV_NAME = "FetchReach-v1"
MAX_EPISODE_STEPS = 100 # Time limit for the episodes
TIMESTEPS = int(3e4)
model_path = "sb3td3_30000s.pth"

env = gym.make(ENV_NAME).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True


# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=MAX_EPISODE_STEPS,
    ),
    verbose=1,
)
# model.buffer_size default to be 1e6

# Train the model
model.learn(TIMESTEPS)
model.save(model_path)

# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load('./her_bit_env', env=env)

obs = env.reset()
env.render()

for i in range(10):
	done = False
	obs = env.reset()
	# while not done:
	for i in range(10):
		action, _state = model.predict(obs, deterministic=True)
		time.sleep(0.5)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			print(info)
			time.sleep(0.3)
			break
	# time.sleep(1)