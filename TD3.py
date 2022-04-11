"""
This is from EECS598-002 hw8.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim
import time

"""
Structure of Actor Network (Action).
"""
class Actor(nn.Module):
    def __init__(self,state_size, action_size, action_bound, fc_units = 256):
        super().__init__()
        self.max_action = action_bound
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        Args:
            state: torch.Tensor with shape (batch_size, state_size)
        Returns:
            action: torch.Tensor with shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # each action of [-self.max_action, self.max_action]
        action = torch.tanh(self.fc3(x)) * self.max_action 
        return action

"""
Structure of Critic Network.
"""
class CriticQ(nn.Module):
    def __init__(self, state_size, action_size, fc_units = 256):
        """
        Args:
            state_size: state dimension
            action_size: action dimension
            fc_units: number of neurons in one fully connected hidden layer
        """
        super().__init__()
        
        # Q (s, a)
        # Q-network 1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc_units)
        self.l2 = nn.Linear(fc_units, fc_units)
        self.l3 = nn.Linear(fc_units, 1)

        # Q-network 2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc_units)
        self.l5 = nn.Linear(fc_units, fc_units)
        self.l6 = nn.Linear(fc_units, 1)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps state-action pairs -> Q-values.
        Args:
            state: torch.Tensor with shape (batch_size, state_size)
            action: torch.Tensor with shape (batch_size, action_size)
        Returns:
            Q_value_1: torch.Tensor with shape (batch_size, 1)
            Q_value_2: torch.Tensor with shape (batch_size, 1)
        """
        state_action = torch.cat([state, action], 1)
        
        x1 = F.relu(self.l1(state_action))
        x1 = F.relu(self.l2(x1))
        Q_value_1 = self.l3(x1)
        
        x2 = F.relu(self.l4(state_action))
        x2 = F.relu(self.l5(x2))
        Q_value_2 = self.l6(x2)
        
        return Q_value_1, Q_value_2

"""
Implementation of TD3 Algorithm
"""
class TD3:
    def __init__(self, state_size, action_size, action_upper_bound, action_lower_bound):
        self.lr_actor = 1e-3  # learning rate for actor network
        self.lr_critic = 1e-3  # learning rate for critic network
        self.buffer_capacity = 100000  # replay buffer capacity
        self.batch_size = 128  # mini-batch size
        self.tau = 0.02  # soft update parameter
        self.policy_delay = 2  # policy will be updated once every policy_delay times for each update of the Q-networks.
        self.gamma = 0.99  # discount factor
        self.target_noise = 0.2  # standard deviation for smoothing noise added to target policy
        self.noise_clip = 0.5  # limit for absolute value of target policy smoothing noise.
        self.update_every = 200  # number of env interactions that should elapse between updates of Q-networks.
        # Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps should be 1.
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda") # torch.device("cpu")
        self.action_upper_bound = action_upper_bound  # action space upper bound
        self.action_lower_bound = action_lower_bound  # action space lower bound
        self.create_actor()
        self.create_critic()
        self.act_opt = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.replay_memory_buffer = deque(maxlen=self.buffer_capacity)
        
    def create_actor(self):
        self.actor = Actor(self.state_size, self.action_size, self.action_upper_bound).to(self.device) # for main network
        self.actor_target = Actor(self.state_size, self.action_size, self.action_upper_bound).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def create_critic(self):
        self.critic = CriticQ(self.state_size, self.action_size).to(self.device) # for main network
        self.critic_target = CriticQ(self.state_size, self.action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        """
        Add samples to replay memory
        Args:
            state: current state, a numpy array with shape (state_size,)
            action: current action, a numpy array with shape (action_size,)
            reward: reward obtained
            next_state: next state, a numpy array with shape (state_size,)
            done: True when the current episode ends, False otherwise
        """
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self):
        """
        Random samples from replay memory without replacement
        Returns a self.batch_size length list of unique elements chosen from the replay buffer.
        Returns:
            random_sample: a list with len=self.batch_size,
                           where each element is a tuple (state, action, reward, next_state, done)
        """
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample
    
    def soft_update_target(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train(self, cur_time_step, episode_time_step, state, action, reward, next_state, done):
        """
        Collect samples and update actor network and critic network using mini-batches of experience tuples.
        Args:
            cur_time_step: current time step counting from the beginning, 
                           which is equal to the number of times the agent interacts with the environment
            episode_time_step: the time step counting from the current episode
            state: current state, a numpy array with shape (state_size,)
            action: current action, a numpy array with shape (action_size,)
            reward: reward obtained
            next_state: next state, a numpy array with shape (state_size,)
            done: True when the current episode ends, False otherwise
        """
        self.add_to_replay_memory(state, action, reward, next_state, done)      
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        if cur_time_step % self.update_every != 0: # We update once every 200 environment interations
            return
        
        # Perform self.update_every times of updates of the critic networks and 
        # (self.update_every / policy_delay) times of updates of the actor network
        for it in range(self.update_every): 
            """
            state_batch: torch.Tensor with shape (self.batch_size, state_size), a mini-batch of current states
            action_batch: torch.Tensor with shape (self.batch_size, action_size), a mini-batch of current actions
            reward_batch: torch.Tensor with shape (self.batch_size, 1), a mini-batch of rewards
            next_state_batch: torch.Tensor with shape (self.batch_size, state_size), a mini-batch of next states
            done_list: torch.Tensor with shape (self.batch_size, 1), a mini-batch of 0-1 integers, 
                   where 1 means the episode terminates for that sample;
                         0 means the episode does not terminate for that sample.
            """
            mini_batch = self.get_random_sample_from_replay_mem()
            state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)
            
            # Please complete codes for updating the critic networks
            """
              Make sure to consider whether the corresponding episode terminates when calculating target values.
                If the episode terminates, then the next state value should be 0.
            """
            ### BEGIN SOLUTION
            # YOUR CODE HERE
            # Train Critic
            target_action_batch = self.actor_target(next_state_batch)
            noise = (torch.randn_like(target_action_batch) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            clipped_action = (target_action_batch + noise).clamp(self.action_lower_bound, self.action_upper_bound)

            target_Q_value_1, target_Q_value_2 = self.critic_target(next_state_batch, clipped_action)
            target_Q = torch.minimum(target_Q_value_1, target_Q_value_2)
            y = reward_batch + (1 - done_list) * self.gamma * target_Q

            Q_value_1, Q_value_2 = self.critic(state_batch, action_batch)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(Q_value_1, y) + loss_fn(Q_value_2, y)
            
            self.crt_opt.zero_grad()
            loss.backward()
            self.crt_opt.step()
            ### END SOLUTION
           
            # Train Actor
            # Delayed policy updates
            # Update self.actor once every policy_delay times for each update of self.critic
            if it % self.policy_delay == 0:
                
                # Please complete codes for updating of the actor network
                """
                Hint: 
                  You may update self.actor using the optimizer self.act_opt and recall the loss function for DDPG training
                """
                ### BEGIN SOLUTION
                # YOUR CODE HERE
                best_action_batch = self.actor(state_batch)
                Q1, Q2 = self.critic(state_batch, best_action_batch)
                loss = - Q1.mean()

                self.act_opt.zero_grad()
                loss.backward()
                self.act_opt.step()
                
                ### END SOLUTION
                
                # Soft update target models
                self.soft_update_target(self.critic, self.critic_target)
                self.soft_update_target(self.actor, self.actor_target)
            
    
    def policy(self, state):
        """
        Select action based on the actor network.
        Args:
            state: a numpy array with shape (state_size,)
        Returns:
            actions: a numpy array with shape (action_size,)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = np.squeeze(self.actor(state).cpu().data.numpy())
        self.actor.train()
        # return actions[None]
        return actions