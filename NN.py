import random
import numpy as np
from numpy import asarray
from matplotlib import pyplot
import matplotlib.pyplot as plt

import torch
from torch import float32, nn, tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from Parameter import *
import time
import os

def normalization(X, din_Max=0, din_Min=0, Max=1,Min=0):
    if np.all(din_Max == 0) or np.all(din_Min == 0):
        data_MAX = X.max(axis=0)
        data_MIN = X.min(axis=0)
    else:
        data_MAX = din_Max
        data_MIN = din_Min
    x_std = (X - data_MIN)/(data_MAX - data_MIN +10e-15)
    X_scaled = x_std*(Max - Min+10e-15) + Min
    return X_scaled, data_MAX, data_MIN


def denormalization(X_scaled,data_Max, data_MIN,Max=1,Min=0):
    Y = (X_scaled - Min)/(Max - Min)
    X = Y * (data_Max - data_MIN) + data_MIN
    return X

# Building Our Mode
class Network(nn.Module):
    # Declaring the Architecture
    def __init__(self,input_dim, hidden_dim, output_dim,seed=0):
        super(Network,self).__init__()
        torch.manual_seed(seed)

        hidden_dim1 = hidden_dim[0]
        hidden_dim2 =hidden_dim[1]
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        nn.init.uniform_(self.fc1.weight, -1.5, 1.5)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        nn.init.uniform_(self.fc2.weight, -1.5, 1.5)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        nn.init.uniform_(self.fc3.weight, -1.5, 1.5)



    # Forward Pass
    def forward(self, x):
        h1 = self.fc1(x)
        h1 = F.relu(h1)
        h2 = self.fc2(h1)
        h2 = F.relu(h2)
        y = self.fc3(h2)
        return y


def training_NN(Zs, Ys, use_cuda=False, epochs=10,
                            lr=0.1, thr=0, use_ard=False, composite_kernel=False,
                            ds=None, global_hyperparams=False,
                            model_hyperparams=None):
    ###############################################################################
    #                  Create the training data
    ###############################################################################
    num_data = 5000
    num_batch = 500

    input_state, out_state = create_data(num_data,MODE=1,output_mode=input_para)
    input_normed,test_in_max,test_in_min = normalization(input_state) 
    out_normed, test_out_max, test_out_min = normalization(out_state)
    
    input_tensor = torch.tensor(input_normed, dtype=float32)
    out_tensor = torch.tensor(out_normed,dtype=float32)
    tra_data_set = TensorDataset(input_tensor,out_tensor)
    trainloader = DataLoader(tra_data_set, batch_size=num_batch)
    # Validation data 
    input_V_state, out_V_state = create_data(num_data,MODE=1,seed_v=10,output_mode=input_para) 
    input_V_normed = normalization(input_V_state)[0]
    out_V_normed = normalization(out_V_state)[0]
    
    input_V_tensor = torch.tensor(input_V_normed,dtype=float32)
    out_V_tensor = torch.tensor(out_V_normed,dtype=float32)
    val_data_set = TensorDataset(input_V_tensor,out_V_tensor)
    valiloader = DataLoader(val_data_set, batch_size=num_batch)


    ###############################################################################
    #              Define the network
    ###############################################################################
    input_dim = input_state.shape[1]
    hidden_dim = [100,100]
    output_dim = 3
   

   
    # Training with Validation
    epochs = 1000

    start = time.time()
    path = '/home/yatin/Documents/'
    for i in range(1):
        mypath = path + 'bin' + str(i)
        if not os.path.isdir(mypath):
            os.makedirs(mypath)
        model = Network(input_dim,hidden_dim,output_dim,i)
        if torch.cuda.is_available():
            model = model.cuda()
        min_valid_loss = np.inf
        loss_list =[]
        val_loss_list = []
        # Declaring Loss and Optimizer
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas=(0.9, 0.999)) #adam

        for e in range(epochs):
            train_loss = 0.0
            for data, labels in trainloader:
                # Transfer Data to GPU if available
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                
                # Clear the gradients
                optimizer.zero_grad()
                # Forward Pass
                target = model(data)
                # Find the Loss
                loss = criterion(target,labels)
                # Calculate gradients
                loss.backward()
                # Update Weights
                optimizer.step()
                # Calculate Loss
                train_loss += loss.item()
            loss_list.append(train_loss)
            
            valid_loss = 0.0
            model.eval()	 # Optional when not using Model Specific layer
            for data, labels in valiloader:
                # Transfer Data to GPU if available
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                
                # Forward Pass
                target = model(data)
                # Find the Loss
                loss = criterion(target,labels)

                # Calculate Loss
                valid_loss += loss.item()
            val_loss_list.append(valid_loss)
            print(f'Epoch {e+1} \t\t Training Loss:{train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(valiloader)}')
            
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        
                min_valid_loss = valid_loss
                save_num = e
                # Saving State Dict
                torch.save(model.state_dict(), mypath + '/saved_model.pth')

        np.save(mypath + '/loss.npy',loss_list)
        np.save(mypath + '/va_loss.npy',val_loss_list)
        print("save model number is : ",save_num)
        print("test data input max", test_in_max)
        print("test data input min", test_in_min)
        print("test data out put max", test_out_max)
        print("test data out put min", test_out_min)
        scaling = [test_in_max, test_in_min, test_out_max, test_out_min]
        np.save(mypath + '/scaling.npy',scaling)


