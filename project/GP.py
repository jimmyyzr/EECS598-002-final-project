# GPyTorch Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import time
from sklearn.metrics import mean_squared_error as mse
import gpytorch
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MultitaskKernel
from gpytorch.kernels import RBFKernel, RBFKernel, ProductKernel
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

# Math, avoiding memory leak, and timing
import math
import gc
import math

#My Imports
import numpy as np
from Parameter import *

# PyTorch Imports
import torch
from Parameter import *
from scipy.stats import multivariate_normal

def normalization(X, din_Max=0, din_Min=0, Max=1,Min=0):
    if np.all(din_Max == 0) or np.all(din_Min == 0):
        data_MAX = X.max(axis=0)
        data_MIN = X.min(axis=0)
    else:
        data_MAX = din_Max
        data_MIN = din_Min
    x_std = (X - data_MIN)/(data_MAX - data_MIN + 10e-15)
    X_scaled = x_std*(Max - Min) + Min
    return X_scaled, data_MAX, data_MIN


def denormalization(X_scaled,data_Max, data_MIN,Max=1,Min=0):
    Y = (X_scaled - Min)/(Max - Min)
    X = Y * (data_Max - data_MIN) + data_MIN
    return X

class BatchedGP(ExactGP):
    """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training.
    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) XD is the dimension of the features (d_state + d_action)
                (iv) YD is the dimension of the labels (d_reward + d_state)
            The features of train_x are tiled YD times along the first dimension.
        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) YD is the dimension of the labels (d_reward + d_state)
            The features of train_y are stacked.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        output_device (str):  The device on which the GPR will be trained on.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
    """
    def __init__(self, train_x, train_y, likelihood, shape, output_device, use_ard=False):

        # Run constructor of superclass
        super(BatchedGP, self).__init__(train_x, train_y, likelihood)

        # Determine if using ARD
        ard_num_dims = None
        if use_ard:
            ard_num_dims = train_x.shape[-1]

        # Create the mean and covariance modules
        self.shape = torch.Size([shape])
        self.mean_module = ConstantMean(batch_shape=self.shape)
        self.base_kernel = RBFKernel(batch_shape=self.shape,
                                        ard_num_dims=ard_num_dims)
        self.covar_module = ScaleKernel(self.base_kernel,
                                        batch_shape=self.shape,
                                        output_device=output_device)

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.
        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.
        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultivariateNormal(mean_x, covar_x)




def train_gp_batched_scalar(Zs, Ys, use_cuda=False, epochs=10,
                            lr=0.1, thr=0, use_ard=False, composite_kernel=False,
                            ds=None, global_hyperparams=False,
                            model_hyperparams=None):
    """Computes a Gaussian Process object using GPyTorch. Each outcome is
    modeled as a single scalar outcome.
    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (np.array): Array of predicted values of shape (B, N, YD), where B is the
            size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        use_cuda (bool): Whether to use CUDA for GPU acceleration with PyTorch
            during the optimization step.  Defaults to False.
        epochs (int):  The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float):  The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float):  The mll threshold at which to stop training.  Defaults to 0.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        composite_kernel (bool):  Whether to use a composite kernel that computes
            the product between states and actions to compute the variance of y.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state.  Only applicable if composite_kernel is True.
        global_hyperparams (bool):  Whether to use a single set of hyperparameters
            over an entire model.  Defaults to False.
        model_hyperparams (dict):  A dictionary of hyperparameters to use for
            initializing a model.  Defaults to None.
    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Preprocess batch data
    B, N, XD = Zs.shape
    YD = Ys.shape[-1]
    batch_shape = B * YD

    if use_cuda:  # If GPU available
        output_device = torch.device('cuda:0')  # GPU
    else:
        output_device = torch.device('cpu')

    # Format the training features - tile and reshape
    train_x = torch.tensor(Zs, device=output_device)
    train_x = train_x.repeat((YD, 1, 1))

    # Format the training labels - reshape
    train_y = torch.vstack(
        [torch.tensor(Ys, device=output_device)[..., i] for i in range(YD)])

    # initialize likelihood and model
    likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_shape]))

    # Determine which type of kernel to use

   
    model = BatchedGP(train_x, train_y, likelihood, batch_shape,
                        output_device, use_ard=use_ard)

    # Initialize the model with hyperparameters
    if model_hyperparams is not None:
        model.initialize(**model_hyperparams)

    # Determine if we need to optimize hyperparameters
    if global_hyperparams:
        if use_cuda:  # Send everything to GPU for training
            model = model.cuda().eval()

            # Empty the cache from GPU
            torch.cuda.empty_cache()
            gc.collect()  # NOTE: Critical to avoid GPU leak
            del train_x, train_y, Zs, Ys, likelihood

        return model, model.likelihood

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if use_cuda:  # Send everything to GPU for training
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        mll = mll.cuda()

    def epoch_train(j):
        """Helper function for running training in the optimization loop.  Note
        that the model and likelihood are updated outside of this function as well.
        Parameters:
            j (int):  The epoch number.
        Returns:
            item_loss (float):  The numeric representation (detached from the
                computation graph) of the loss from the jth epoch.
        """
        optimizer.zero_grad()  # Zero gradients
        output = model(train_x)  # Forwardpass
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backpropagate gradients
        item_loss = loss.item()  # Extract loss (detached from comp. graph)
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Zero gradients
        gc.collect()  # NOTE: Critical to avoid GPU leak
        return item_loss

    # Run the optimization loop
    for i in range(epochs):
        loss_i = epoch_train(i)
        if i % 20 == 0:
            print("LOSS EPOCH {}: {}".format(i, loss_i))
        if loss_i < thr:  # If we reach a certain loss threshold, stop training
            break

    # Empty the cache from GPU
    torch.cuda.empty_cache()

    return model, likelihood


def preprocess_eval_inputs(Zs, d_y, device="cpu"):
    """Helper function to preprocess inputs for use with training
    targets and evaluation.
    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        d_y (int):  The dimensionality of the targets of GPR.
        device (str):  Device to output the tensor to.
    Returns:
        eval_x (torch.tensor):  Torch tensor of shape (B * YD, N, XD).  This
            tensor corresponding to a tiled set of inputs is used as input for
            the inference model in FP32 format.
    """
    # Preprocess batch data
    eval_x = torch.tensor(Zs, device=device).double()
    eval_x = eval_x.repeat((d_y, 1, 1))
    return eval_x

######################################################################
def gp_train(train_x, train_y):
  
    train_x_norm,din_max,din_min = normalization(train_x) 
    train_y_norm, dout_max, dout_min = normalization(train_y)
    train_x_norm = train_x_norm[None, ...]
    train_y_norm = train_y_norm[None, ...]
    model, likelihood = train_gp_batched_scalar(train_x_norm, train_y_norm,
                                        use_cuda=USE_CUDA,
                                        composite_kernel=COMPOSITE_KERNEL,
                                        epochs=EPOCHS, lr=LR, thr=THR, ds=Ds,
                                        use_ard=USE_ARD)
    scale = [din_max, din_min, dout_max, dout_min]
    gp_model = [model,likelihood,scale]

    return gp_model


def gp_eval(test_x, model, likelihood, scale):
    model.eval()
    likelihood.eval()
    din_max, din_min, dout_max, dout_min = scale[:]
    test_x_norm = normalization(test_x,din_max,din_min)[0]
    test_x_norm = test_x_norm[None, ...]
    test_x = preprocess_eval_inputs(test_x_norm, DY)
    if USE_CUDA:
        test_x = test_x.cuda()
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        variance = observed_pred.variance
    output = mean.cpu().detach().numpy()
    predic_test = np.squeeze(np.array([output[i::B] for i in range(B)])).T
    predic_test  = denormalization(predic_test,dout_max,dout_min)
    return predic_test




















