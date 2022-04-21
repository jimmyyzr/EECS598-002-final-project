import numpy as np
from GP import gp_train, gp_eval
import random


class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self):
        self.data_x = []
        self.data_y = []
        self.batch_size = 500
        self.model = None

    def store_data(self, s, a, r, s_next):
        """
        s: np.ndarray, shape (X, 10)
        a: np.ndarray, shape (X, 4)
        r: np.ndarray, shape (X, 1)
        s_next: np.ndarray, shape (X, 10)
        """
        x = np.concatenate((s, a), axis=1)
        self.data_x.append(x.tolist()) # TODO: why convert np.ndarray to list, you will convert it back later
        y = np.concatenate((r, s_next), axis=1)
        self.data_y.append(y.tolist())
    
    def random_sampling(self):
        # # delete 1/5th of the buffer when full
        # if self.size > self.max_size:
        #     del self.buffer[0:int(self.size/5)]
        #     self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.data_x), size=self.batch_size)
        input, output = [], []
        
        for i in indexes:
            input.append(np.array(self.data_x[i], copy=False))
            output.append(np.array(self.data_y[i], copy=False))

        
        return np.array(input), np.array(output)

    def update_model(self):
        train_x,train_y = self.random_sampling()

        print(type(train_x))
        print(train_x.shape,train_y.shape)
        exit(0)
        self.model = gp_train(train_x, train_y)
    # generate next state by simulation
    def step(self, s, a):
        test_x = np.concatenate((s, a)).reshape(1,-1)
        model,likelihood,scale = self.model[:]
        predic =  gp_eval(test_x, model, likelihood, scale)
        r = predic[0]
        s_next = predic[1:]
        return r, s_next



class EnvModel_NN:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self):
        self.data_x = []
        self.data_y = []
        self.batch_size = 500
        self.model = None

    def store_data(self, s, a, r, s_next):
        
        x = np.concatenate((s, a))
        self.data_x.append(x.tolist())
        r = np.array([r])
        y = np.concatenate((r, s_next))
        self.data_y.append(y.tolist())
        

    def update_model(self):
        random_sample_x = random.sample(self.data_x, self.batch_size)
        random_sample_y = random.sample(self.data_y, self.batch_size)

        train_x = np.array(self.data_x)
        train_y = np.array(self.data_y)
        self.model = gp_train(train_x, train_y)
    # generate next state by simulation
    def step(self, s, a):
        test_x = np.concatenate((s, a)).reshape(1,-1)
        model,likelihood,scale = self.model[:]
        predic =  gp_eval(test_x, model, likelihood, scale)
        r = predic[0]
        s_next = predic[1:]
        return r, s_next
