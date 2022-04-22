import matplotlib.pyplot  as plt
import numpy as np
import matplotlib as mpl

a = np.load("ep_reward_list_mpc.npy")
b = np.load("ep_reward_list_td3.npy")
# c = np.load("ep_reward_td3_UVFA_dense.npy")
# mpl.style.use(sty)
print("hhhh")
fig1=  plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
episode = np.arange(0,len(a))
ax1.plot(episode,a,label='MPC actor')
ax1.plot(episode,b,label='TD3 actor',color = '#ff7f0e')
ax1.title.set_text('Reward vs episode')
ax1.set_xlabel('episode')
ax1.set_ylabel('Reward')
ax1.legend()

plt.show()