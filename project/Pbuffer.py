# P buffer is only temporary
# We just need to obeu its .sample() return format

# Always check if self._vec_normalize_env is None
# It controls unnormalize in _store_transition and nomalize in buffer.sample()

import torch
from stable_baselines3.common.type_aliases import DictReplayBufferSamples # just NamedTuple
from typing import Dict


class Pbuffer():
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.obs = None
        self.action = None
        self.next_obs = None
        self.reward = None
        self.dones = None

    # assumed input are torch.tensor
    # done really means reaching the goal
    def add(self,
            obs: Dict[str, torch.tensor],
            next_obs: Dict[str, torch.tensor],
            action: torch.tensor,
            reward: torch.tensor,
            done: torch.tensor) -> None:
        self.obs = obs
        self.next_obs = next_obs
        self.action = action
        self.reward = reward
        self.dones = done


    def sample(self, batch_size, env): # arguments are ignored, just for compatibility
        assert env is None
        return DictReplayBufferSamples(
            observations=self.obs,
            next_observations=self.next_obs,
            actions=self.action,
            dones=self.dones,
            rewards=self.reward
        )