import torch

"""Main parameters for GP."""
# Set parameters
B = 1  # Number of batches
# N = 100  # Number of data points in each batch
DY = 4  # Dimension Y data
Ds = 1  # Dimensions for first factored kernel - only needed if factored kernel is used
EPOCHS = 20  # Number of iterations to perform optimization
THR = -1e5  # Training threshold (minimum)
USE_CUDA = torch.cuda.is_available()  # Hardware acceleraton
MEAN = 0  # Mean of data generated
SCALE = 1  # Variance of data generated
COMPOSITE_KERNEL = False  # Use a factored kernel
USE_ARD = True  # Use Automatic Relevance Determination in kernel
LR = 0.5  # Learning rate