import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import shutil
from fund_rl.agents.base import TAgent
import numpy as np
from gymnasium.wrappers import FrameStackObservation

def Make_Environment(Environment: str, **kwargs):
    """
    Create and return a Gymnasium environment.

    :param Environment: The name of the environment to create (e.g., ``CartPole-v1``).
    :type Environment: str

    :param kwargs: Additional keyword arguments passed to :func:`gym.make`.
    :type kwargs: Any

    :return: Instantiated Gymnasium environment.
    :rtype: gym.Env
    """
    return gym.make(Environment, **kwargs)

class TRecorder(gym.Wrapper):
    """
    A Gym environment wrapper that records videos of episodes during training or evaluation.

    Videos are saved in a structured directory based on agent name, environment ID, 
    and whether the agent is in training or evaluation mode. The folder is cleared on initialization.

    Args:
        Environment (gym.Env): The Gym environment to wrap and record.
        Agent (TAgent): The agent interacting with the environment. Used to determine recording mode and naming.
        Folder_Name (str, optional): Base folder name to save videos in. Defaults to 'Videos'.
        Record_Every_N (int, optional): Frequency (in episodes) at which to record. Defaults to 1 (record every episode).
    """
    def __init__(self, Environment : gym.Env, Agent : TAgent, Folder_Name: str = 'Videos', Record_Every_N: int = 1):

        ls_Prefix = "Training" if Agent.Is_Training else "Evaluation"
        ls_Folder = Folder_Name + "/" + Agent.Name + "/" + Environment.spec.id + "/" + ls_Prefix
        if os.path.exists(ls_Folder):
            shutil.rmtree(ls_Folder)

        Environment = RecordVideo(Environment, video_folder=ls_Folder, name_prefix=ls_Prefix, episode_trigger=lambda x: x % Record_Every_N == 0)
        super().__init__(Environment)


class TOne_Hot_Encode(gym.ObservationWrapper):
    """
    A Gym ObservationWrapper that converts discrete observations into one-hot encoded vectors.

    This wrapper transforms a scalar discrete observation into a one-hot encoded NumPy array,
    which can be more suitable for neural network inputs in certain environments.

    Only supports environments with a Discrete observation space.

    Args:
        Environment (gym.Env): The Gym environment to wrap. Must have a Discrete observation space.
    """
    def __init__(self, Environment : gym.Env):
        super(TOne_Hot_Encode, self).__init__(Environment)
        if not isinstance(Environment.observation_space, gym.spaces.Discrete):
            raise TypeError("TOneHotEncode only supports Discrete observation spaces.")
        self.gi_N = Environment.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1, shape=(self.gi_N,), dtype=np.float32)

    def observation(self, Observation):
        larr_One_Hot = np.zeros(self.gi_N, dtype=np.float32)
        larr_One_Hot[Observation] = 1.0
        return larr_One_Hot

class TSelect_Observations(gym.ObservationWrapper):
    """
    A Gym ObservationWrapper that selects and returns only a subset of the observation vector.

    This is useful when only certain dimensions of the environment's observation space are relevant 
    for the agent or experiment.

    Args:
        Environment (gym.Env): The Gym environment to wrap.
        Indices (list[int]): The indices of the observation vector to keep.
    """
    def __init__(self, Environment : gym.Env, Indices : list[int]):
        super().__init__(Environment)
        self.garr_Indices = Indices

        low = self.observation_space.low[self.garr_Indices]
        high = self.observation_space.high[self.garr_Indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def observation(self, Observation):
        return Observation[self.garr_Indices]


class TAtari_Normalized_Environment(gym.ObservationWrapper):
    """
    A Gym ObservationWrapper that normalizes Atari-style image observations to the [0, 1] range.

    This wrapper scales raw 8-bit pixel observations (0–255) by dividing by 255.0, which is a common
    preprocessing step for neural networks to improve training stability and performance.
    """
    def observation(self, Observation):
        return (Observation / 255.0)

class TAtari_Pong_RAM_Environment(gym.ObservationWrapper):
    """
    A Gym ObservationWrapper for Atari Pong using RAM-based observations.

    This wrapper extracts a compact and relevant 6-byte subset from the full 128-byte
    Atari RAM observation, specifically targeting key entities like the player paddle,
    opponent paddle, and the ball.

    The selected bytes represent:
        - Player Y position  (RAM[51])
        - Player X position  (RAM[46])
        - Enemy Y position   (RAM[50])
        - Enemy X position   (RAM[45])
        - Ball X position    (RAM[49])
        - Ball Y position    (RAM[54])

    This simplification can reduce input dimensionality and speed up training.

    Args:
        Environment (gym.Env): The original Atari Pong environment with RAM observations.
    """
    def __init__(self, Environment : gym.Env):
        super().__init__(Environment)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)

    def observation(self, Observation):
        # Select relevant bytes from the 128-byte RAM
        larr_Small_Observation = np.array([
            Observation[51],  # Player Y
            Observation[46],  # Player X
            Observation[50],  # Enemy Y
            Observation[45],  # Enemy X
            Observation[49],  # Ball X
            Observation[54],  # Ball Y
        ], dtype=np.uint8)
        return larr_Small_Observation


class TFramestack(FrameStackObservation):
    """
    A Gym Wrapper that stacks multiple frames from the environment's observation space.

    This is useful for environments where temporal context is important, such as in video games,
    where the agent benefits from seeing a sequence of frames rather than just the current frame.

    Args:
        Environment (gym.Env): The Gym environment to wrap.
        Stack_Size (int): The number of frames to stack together.
    """
    def __init__(self, Environment : gym.Env, Stack_Size : int = 4) -> None:
        super().__init__(Environment, Stack_Size)

class TFlatten_Observation(gym.ObservationWrapper):
    """
    A Gym ObservationWrapper that flattens the observation into a 1D array.

    Useful for feeding data into networks that expect vector inputs.
    """
    def __init__(self, env):
        super().__init__(env)
        original_shape = env.observation_space.shape
        original_dtype = env.observation_space.dtype

        flat_dim = int(np.prod(original_shape))
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(flat_dim,),
            dtype=original_dtype
        )

    def observation(self, Observation):
        return Observation.flatten()

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, Environment, Skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, Environment)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+Environment.observation_space.shape, dtype=np.uint8)
        self._skip = Skip

    def step(self, Action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(Action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)