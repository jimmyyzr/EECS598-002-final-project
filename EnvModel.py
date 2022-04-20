import numpy as np
from GP import gp_train, gp_eval


class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self):
        self.data_x = []
        self.data_y = []
        self.model = None

    def store_data(self, s, a, r, s_next):
        
        x = np.concatenate((s, a))
        self.data_x.append(x.tolist())
        r = np.array([r])
        y = np.concatenate((r, s_next))
        self.data_y.append(y.tolist())
        

    def update_model(self):
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
