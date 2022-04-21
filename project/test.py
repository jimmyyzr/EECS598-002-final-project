import numpy as np

pos1 = 0
pos2 = 0
buffer_size = 10
x = np.arange(pos1, pos2 + buffer_size) % buffer_size

print(x)