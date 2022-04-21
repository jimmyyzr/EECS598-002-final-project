import matplotlib.pyplot  as plt
import numpy as np

a = np.load("ep_reward_list_mpc.npy")
b = np.load("ep_reward_list_td3.npy")

print("hhhh")
fig1=  plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
episode = np.arange(0,len(a))
ax1.plot(episode,a)
plt.show()