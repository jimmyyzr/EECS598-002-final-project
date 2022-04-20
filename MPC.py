from random import seed
import numpy as np
import gym
import copy
class MPCController:
    def __init__(self,H,K,env,dynamic_model):
        self.NUM = K # number of K candidate action sequences 
        self.H = H # finite horizon H
        self.env = env #real env
        self.dyna_model = dynamic_model #learned dynamics model, input: s0,a0 output s1

        pass

    
    def cost_function(self,s,a,s_next,g):
        cost = np.sum(np.sqrt((g-s_next[0:3])**2))
        return cost
    
    def get_best_action(self,s_inital,goal):
    #   generate_control_sequences
        lowbound = self.env.action_space.low
        upperbound =self.env.action_space.high
        action_dim = self.env.action_space.shape[0]
        all_control_samples = np.random.uniform(lowbound, upperbound, (self.NUM, self.H,action_dim ))
    
    #   H horizon extend
        whole_cost_list = []
        for u_sequence in all_control_samples:
            s_t = s_inital
            traj_list = []
            reward_list = []
            # self.dyna_model.reset(seed=0) #for env
            for a in u_sequence: 
                # s_next_dic, reward, done, info = self.dyna_model.step(a)  #for env 
                # s_next = s_next_dic["observation"]  #for env
                r,s_next = self.dyna_model.step(s_t,a) 

                c_t = self.cost_function(s_t,a,s_next,goal)
                traj_list.append(s_t) 
                reward_list.append(c_t)
                s_t = s_next
            whole_cost_list.append(reward_list)

    #   Choose best action
        cost_arr = np.array(whole_cost_list)
        cost_sum = np.sum(cost_arr,axis=1)
        minmum_cost_index = np.argmin(cost_sum)
        best_action = all_control_samples[minmum_cost_index][0]
        
        return best_action
 

    

def main():
    max_steps = 100
    env_name ="FetchReach-v1"
    env = gym.make(env_name).unwrapped 
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps+1)
    mpc = MPCController(10,100,env,env)

    state_dic = env.reset(seed=0)
    state = state_dic["observation"]
    goal = state_dic["desired_goal"][0:3]

    
    paths = mpc.get_best_action(state,goal)




if(__name__ == '__main__'):
    main()         
         
         