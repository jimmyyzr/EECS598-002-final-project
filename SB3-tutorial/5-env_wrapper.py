import gym

class TimeLimitWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  :param max_steps: (int) Max number of steps per episode
  """
  def __init__(self, env, max_steps=100):
    # Call the parent constructor, so we can access self.env later
    super(TimeLimitWrapper, self).__init__(env)
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    return obs, reward, done, info


# Note that reaching max_episode_steps also returns Done == True

# Method 1
from gym.envs.classic_control.pendulum import PendulumEnv

# Here we create the environment directly because 
# gym.make() already wrap the environement in a TimeLimit wrapper otherwise !!!!!!!!! <---
env = PendulumEnv()
# Wrap the environment
env = TimeLimitWrapper(env, max_steps=100) # use custom wrapper

# Method 2

# add .unwrapped so there is no TimeLimit wrapper when gym.make()
# gym.make("Pendulum-v1") has defult max_episode_steps is 50, try env.spec to see it
# env = gym.make("Pendulum-v1").unwrapped 
# env = gym.wrappers.TimeLimit(env, max_episode_steps=1000) # use gym wrapper

obs = env.reset()
done = False
n_steps = 0
while not done:
  # Take random actions
  random_action = env.action_space.sample()
  obs, reward, done, info = env.step(random_action)
  n_steps += 1

print(n_steps, info)
# Both reach time limit before reaching the terminal state
# Method 1
# 100 {'time_limit_reached': True}

# Method 2
# 1000 {'TimeLimit.truncated': True}