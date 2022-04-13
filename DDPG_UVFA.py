import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, fc_units = 64):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim + state_dim, fc_units),
                        nn.ReLU(),
                        nn.Linear(fc_units, fc_units),
                        nn.ReLU(),
                        nn.Linear(fc_units, action_dim),
                        nn.Tanh() #[-1,1]
                        )
        # max value of actions
        self.action_bound = action_bound
        # self.offset = offset
        
    def forward(self, state, goal):
        return (self.actor(torch.cat([state, goal], 1)) * self.action_bounds) # + self.offset
        
class Critic(nn.Module):
    """Build a critic (value) network that maps state-action-goal pairs -> Q-values."""
    def __init__(self, state_dim, action_dim, fc_units = 64):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim + action_dim + state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(fc_units, fc_units),
                        nn.ReLU(),
                        nn.Linear(fc_units, 1),
                        nn.Sigmoid() #[0,1]
                        )
        
    def forward(self, state, action, goal):
        # rewards are in range [-1, 0]
        return -self.critic(torch.cat([state, action, goal], 1))
    
class DDPG_UVFA:
    def __init__(self, state_dim, action_dim, action_bounds, lr, gamma):
        
        self.actor = Actor(state_dim, action_dim, action_bounds).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.mseLoss = torch.nn.MSELoss()
        self.gamma = gamma
    
    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # select next action
            next_action = self.actor(next_state, goal).detach()
            
            # Compute target Q-value:
            target_Q = self.critic(next_state, next_action, goal).detach() # no gradient descent for target
            target_Q = reward + (1 - done) * self.gamma * target_Q
            
            # Optimize Critic:
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean() # policy: increase Q(s, actor(s)), loss is set to be its nagative
            
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
        
        
        
        
      
        
        