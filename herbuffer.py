"""
This is basically a simplified version of her (online_sampling) from stable-baselines3
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py
"""
import numpy as np
from typing import Dict
from enum import Enum

GoalSelectionStrategy = Enum('GoalSelectionStrategy', 'FUTURE EPISODE FINAL')

class HerReplayBuffer():
    def __init__(
        self,
        env,# VecEnv,
        buffer_size: int,
        max_episode_length: int,
        n_sampled_goal: int = 4,
        goal_selection_strategy = GoalSelectionStrategy.FUTURE,
        handle_timeout_termination: bool = True,
    ):
        """
        buffer_size: The size of the buffer measured in transitions.
        n_sampled_goal: number of sampled goal for each transition
        """ 

        # maximum steps in episode
        self.max_episode_length = max_episode_length
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (n_sampled_goal + 1))

        self.goal_selection_strategy = goal_selection_strategy
        self.env = env

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = buffer_size // self.max_episode_length
        # idx for time steps when adding transition in one episode
        self.current_idx = 0 
        # Counter to prevent overflow
        self.episode_steps = 0 #TODO: Seems we can also use self.pos to presvent
        # buffer episode pointer
        self.pos = 0
        # Check if buffer is filled
        self.full = False

        # Get shape of observation and goal (usually the same)
        self.obs_shape = self.env.observation_space.spaces["observation"].shape
        self.goal_shape = self.env.observation_space.spaces["achieved_goal"].shape
        self.action_dim = self.env.action_space.shape

        # input dimensions for buffer initialization
        input_shape = {
            "observation": self.obs_shape,
            "achieved_goal": self.goal_shape,
            "desired_goal": self.goal_shape,
            "action": self.action_dim,
            "reward": (1,),
            # "next_obs": (self.env.num_envs,) + self.obs_shape, #TODO: Can we use the next one as next_obeservation
            # "next_achieved_goal": (self.env.num_envs,) + self.goal_shape, 
            # "next_desired_goal": (self.env.num_envs,) + self.goal_shape,
            "done": (1,),
        }
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }

        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)


    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)


    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: Dict[str, np.ndarray]
    ) -> None:
        """Add one transition in the buffer. 
        store_episode is called after all transitions from one episode are added
        """
        # Remove termination signals due to timeout, so done really means reaching the goal
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array(info.get("TimeLimit.truncated", False)))
        else:
            done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        # self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        # self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        # self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]


        # update current episode timestep idx pointer
        self.current_idx += 1
        # update current episode counter
        self.episode_steps += 1

        is_full = self.episode_steps >= self.max_episode_length
        if done or is_full:
            self.store_episode() # Mark finish add all transitions in this episode
        if is_full:            
            self.episode_steps = 0


    # called after finishing storing a complete episode (include many times add() to add transitions)
    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # save the length of newly stored length in episode_lengths
        self.episode_lengths[self.pos] = self.current_idx

        # Current episode is finally all added in the buffer, we now update pointer 
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0


    def sample(self, batch_size: int):
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.
        :param batch_size: Number of element to sample
        :return: Samples.
        """
        return self._sample_transitions(batch_size)

    
    def _sample_transitions(
        self,
        batch_size: int,
    ): #-> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :return: Samples.
        """
        # Select which episodes to use, in total batch_size episodes
        assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
        # Do not sample the episode with index `self.pos` as the episode is invalid
        if self.full:
            episode_indices = (
                np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
            ) % self.n_episodes_stored
        else:
            episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size) # batch_size random episodes

        # A subset of the transitions will be relabeled using HER algorithm
        # Here we relabel the first her_ratio * batch_size 
        her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)] # [0, 1, .., # relabled transitions - 1]
        
        # Select which transitions to use (one transition per sampled episode)
        # ep_lengths for sampled episodes
        ep_lengths = self.episode_lengths[episode_indices]
        # because we need next_achieved_goal, so here transitions_indices exlude the last one
        transitions_indices = np.random.randint(ep_lengths - 1)  
        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}
        transitions["next_achieved_goal"] = self._buffer["achieved_goal"][episode_indices, transitions_indices + 1].copy()

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices] = self.env.compute_reward(
                # We use "achieved_goal" for next transition as "next_achieved_goal" for current transition
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                info = {} #TODO: check if the task env use info to compute reward
            )

        obs = {k: transitions[k] for k in ["observation", "achieved_goal", "desired_goal"]}
        next_obs = {k: self._buffer[k][episode_indices, transitions_indices + 1].copy() \
                for k in ["observation", "achieved_goal", "desired_goal"]}


        return (
            obs,
            transitions["action"],
            transitions["reward"],
            next_obs,
            transitions["done"] # Her transition doesn't affect done
        )
    
    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.
        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            goal_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            goal_indices = np.random.randint(
                transitions_indices[her_indices], self.episode_lengths[her_episode_indices]
            )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            goal_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self._buffer["achieved_goal"][her_episode_indices, goal_indices]


    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

if __name__ == '__main__':
    import gym
    max_episode_steps = 100

    env = gym.make('FetchReach-v1').unwrapped
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset()

    buffer = HerReplayBuffer(env, int(1e3), max_episode_steps)

    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, done, info)
        if done:
            env.reset()

    
    buffer.sample(batch_size=10)
    env.close()