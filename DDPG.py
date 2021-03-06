import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Q(s,mu(a)) = over a, max Q(s,a)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, fc_units = 64):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, fc_units),
                        nn.ReLU(),
                        nn.Linear(fc_units, fc_units),
                        nn.ReLU(),
                        nn.Linear(fc_units, action_dim),
                        nn.Tanh() #[-1,1]
                        )
        # max value of actions
        self.action_bound = torch.tensor(action_bound).to(device)
        
    def forward(self, state):
        return self.actor(state) * self.action_bound #[-action_bound, action_bound]
        
class Critic(nn.Module):
    """Build a critic (value) network that maps state-action pairs -> Q-values."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim + action_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid() #[0,1]
                        )
        
    def forward(self, state, action):
        # rewards are in range [-1, 0]
        return -self.critic(torch.cat([state, action], 1))


class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, lr, gamma):
        
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.mseLoss = torch.nn.MSELoss()
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            obs, action, reward, next_obs, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = np.concatenate((obs["observation"], obs["desired_goal"]),axis=1)
            next_state = np.concatenate((next_obs["observation"], next_obs["desired_goal"]),axis=1)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # select next action
            next_action = self.actor(next_state).detach()
            
            # Compute target Q-value:
            target_Q = self.critic(next_state, next_action).detach() # no gradient descent for target
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
            # Optimize Critic:
            critic_loss = self.mseLoss(self.critic(state, action), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state)).mean() # policy: increase Q(s, actor(s)), loss is set to be its nagative
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))  
        
        
        
        
      
        
        