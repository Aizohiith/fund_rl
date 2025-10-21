import gymnasium as gym
import numpy as np
from typing import Optional

class TMulti_Armed_Bandit_Environment(gym.Env):
    """
    Multi-Armed Bandit Environment
    Args:
        n_arms (int): Number of arms in the bandit.
        walk (bool): If True, the reward probabilities will perform a random walk over time.
    """
    def __init__(self, n_arms, walk = False):
        self.n_arms = n_arms
        self.action_space = gym.spaces.Discrete(self.n_arms)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.probs = np.random.normal(0, 1, n_arms)
        self.Walk = walk
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns an initial observation.
        Args:
            seed (int, optional): Seed for the random number generator.
            options (dict, optional): Additional options for resetting the environment.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if (self.Walk):
            self.probs = self.probs + (np.random.random(self.n_arms) * 2 - 1)/10.0
        return self.observation(), {}

    def observation(self):
        """
        Returns the current observation of the environment.
        """

        return np.array([1], dtype=np.float32) # useless observation, agent has to learn from rewards only


    def step(self, action):
        """
        Takes a step in the environment.
        """
        assert 0 <= action < self.n_arms

        reward = np.random.normal(self.probs[action], 1)
        return self.observation(), reward, True, False, {}

    def render(self):
        """
        Renders the environment.
        """
        print(self.probs)

gym.envs.registration.register(
    id='multi-armed-bandit-v0',
    entry_point=TMulti_Armed_Bandit_Environment,
    max_episode_steps=1,
)