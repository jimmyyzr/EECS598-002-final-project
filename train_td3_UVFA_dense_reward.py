import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim


"""
Some parameters
"""
state_size = 6  # state dimension
action_size = 4  # action dimension
fc_units = 256  # number of neurons in one fully connected hidden layer
action_upper_bound = 1  # action space upper bound
action_lower_bound = -1  # action space lower bound
NOISE_VAR = 0.05


"""
Structure of Actor Network.
"""
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_action = action_upper_bound
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
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


"""
Structure of Critic Network.
"""
class CriticQ(nn.Module):
    def __init__(self):
        """
        Args:
            state_size: state dimension
            action_size: action dimension
            fc_units: number of neurons in one fully connected hidden layer
        """
        super().__init__()
        
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
    def __init__(self):
        self.lr_actor = 1e-3  # learning rate for actor network
        self.lr_critic = 1e-3  # learning rate for critic network
        self.buffer_capacity = 100000  # replay buffer capacity
        self.batch_size = 128  # mini-batch size
        self.tau = 0.02  # soft update parameter
        self.policy_delay = 2  # policy will be updated once every policy_delay times for each update of the Q-networks.
        self.gamma = 0.99  # discount factor
        self.target_noise = 0.05  # standard deviation for smoothing noise added to target policy
        self.noise_clip = 0.5  # limit for absolute value of target policy smoothing noise.
        self.update_every = 200  # number of env interactions that should elapse between updates of Q-networks.
        # Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps should be 1.
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda")  # or self.device = torch.device("cuda")
        self.action_upper_bound = action_upper_bound  # action space upper bound
        self.action_lower_bound = action_lower_bound  # action space lower bound
        self.create_actor()
        self.create_critic()
        self.act_opt = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.replay_memory_buffer = deque(maxlen=self.buffer_capacity)
        
    def create_actor(self):
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def create_critic(self):
        self.critic = CriticQ().to(self.device)
        self.critic_target = CriticQ().to(self.device)
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
        if cur_time_step % self.update_every != 0:
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
            Hints: 
              You may use the above tensors: state_batch, action_batch, reward_batch, next_state_batch, done_list
              You may use self.critic_target and self.actor_target as your target networks
              you may use target policy smoothing techniques with hyperparameters self.target_noise and self.noise_clip
              You may use clipped double Q-learning techniques
              You may update self.critic using the optimizer self.crt_opt and MSE loss function.
              Make sure to consider whether the corresponding episode terminates when calculating target values.
                If the episode terminates, then the next state value should be 0.
            """
            ### BEGIN SOLUTION
            # YOUR CODE HERE
            
            #compute targe actions 
            num = len(mini_batch)
            l = self.action_lower_bound
            h = self.action_upper_bound
            c = self.noise_clip
            noise_var = self.target_noise 
            gamma = self.gamma
     
            mu_target = self.actor_target.forward(next_state_batch).cuda()

            noise = torch.clamp(torch.normal(0,noise_var,size=(num,4)),-c,c).cuda()
            next_action =  torch.clamp(mu_target + noise, l, h )
            
            ###target
            Q_t_1, Q_t_2 = self.critic_target.forward(next_state_batch,next_action)
            min_Q = torch.min(Q_t_1,Q_t_2)
            y  =  reward_batch + gamma*(1-done_list)*min_Q
            y = y.detach()

            Q_1,Q_2 = self.critic.forward(state_batch, action_batch) 
            
            loss_c= F.mse_loss(Q_1,y) + F.mse_loss(Q_2,y)
          
            self.crt_opt.zero_grad()
            loss_c.backward()
            # loss_Q2.backward()
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
                mu_s =  self.actor.forward(state_batch)
                mu_s.cuda()
                Q_phi_1, Q_phi_2 =  self.critic.forward(state_batch, mu_s) 
                Q_phi_1.detach()
                loss =  -torch.sum(Q_phi_1)/num
                
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
        return actions




def main():
    c = 0.3
    timesteps_count = 0  # Counting the time steps
    max_steps = 100  # Maximum time steps for one episode
    ep_reward_list = deque(maxlen=50)
    avg_reward = -9999
    # The following code is provided for the training of your agent in the 'BipedalWalker-v3' gym environment.
    gym.logger.set_level(40)
    env_name ="FetchReach-v1"
    env = gym.make(env_name).unwrapped 
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)
    h = env.action_space.high[0]
    l = -h
    _ = env.reset(seed=0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    


    agent = TD3()

    for ep in range(600):
        state_dic = env.reset()
        desire_goal = state_dic["desired_goal"]
        state = state_dic["observation"][0:3]
        
        episodic_reward = 0
        timestep_for_cur_episode = 0

        for st in range(max_steps):
            # Select action according to policy
            # if ep>300:
            #     env.render()
            s = np.concatenate((state, desire_goal))
            action = agent.policy(s)
            # action = torch.tensor(agent.policy(state)).reshape(1,4)
            # noise = torch.clamp(torch.normal(0,NOISE_VAR,size=(1,4)),-c,c) 
            # action =  np.squeeze(torch.clamp(action + noise, l, h ).cpu().data.numpy())

            # Recieve state and reward from environment.
            next_state_dic, reward, done, info = env.step(action)
            next_state = next_state_dic["observation"][0:3]
            a = np.sum(np.sqrt((desire_goal-next_state)**2))
            reward  = a * reward
            episodic_reward += reward
            s_next = np.concatenate((next_state, desire_goal))
            # Send the experience to the agent and train the agent
            agent.train(timesteps_count, timestep_for_cur_episode, s, action, reward, s_next, done)
            
            timestep_for_cur_episode += 1     
            timesteps_count += 1
            
            # End this episode when `done` is True
            if done:
                break
            state = next_state

        ep_reward_list.append(episodic_reward)
        print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')
        
        if len(ep_reward_list) == 50:
            # Mean of last 50 episodes
            avg_reward = sum(ep_reward_list) / 50
            print(', Moving Average Reward: {:.2f}'.format(avg_reward))
        else:
            print('')

    print('Average reward over 50 episodes: ', avg_reward)
    env.close()

    actor_path = "actor_td3_UVFA.pth"
    torch.save(agent.actor.to("cpu").state_dict(), actor_path)


if(__name__ == '__main__'):
    main()         
         
         