from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import gym

model_class = TD3  # works also with SAC, DDPG and TD3
ENV_NAME = "FetchReach-v1"
MAX_EPISODE_STEPS = 100 # Time limit for the episodes
TIMESTEPS = int(8e4)
model_path = "sb3td3" + str(TIMESTEPS)

env = gym.make(ENV_NAME).unwrapped 
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        online_sampling=True,
        max_episode_length=MAX_EPISODE_STEPS,
    ),
    verbose=1,
)
# model.buffer_size default to be 1e6

# Train the model
from stable_baselines3.common.logger import configure
tmp_path = "sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)


model.learn(TIMESTEPS)
model.save(model_path)