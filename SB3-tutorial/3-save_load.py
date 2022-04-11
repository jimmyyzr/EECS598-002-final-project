import gym
from stable_baselines3 import A2C, SAC, PPO, TD3
import os

# Create save dir
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

model = PPO('MlpPolicy', 'Pendulum-v1', verbose=0).learn(8000)
# The model will be saved under PPO_tutorial.zip
model.save(save_dir + "/PPO_tutorial")

# sample an observation from the environment
obs = model.env.observation_space.sample()

# Check prediction before saving
print("pre saved", model.predict(obs, deterministic=True))

del model # delete trained model to demonstrate loading

loaded_model = PPO.load(save_dir + "/PPO_tutorial")
# Check that the prediction is the same after loading (for the same observation)
print("loaded", loaded_model.predict(obs, deterministic=True))