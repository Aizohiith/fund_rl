from fund_rl.buffers.base import TBuffer
from collections import deque
import random
import numpy as np

class TReplay_Buffer(TBuffer):
    """
    Replay buffer for storing and sampling experiences in reinforcement learning.

    Stores a fixed number of experiences and allows random sampling of batches
    for training.

    Inherits from :class:`TBuffer`.

    :param Memory_Capacity: Maximum number of experiences the buffer can hold.
    :type Memory_Capacity: int

    :param Batch_Size: Number of experiences to sample during training.
    :type Batch_Size: int
    """
    def __init__(self, Memory_Capacity : int = 100_000, Batch_Size : int = 32) -> None:
        """
        Initialize the replay buffer.

        :param Memory_Capacity: Max number of experiences stored.
        :type Memory_Capacity: int
        :param Batch_Size: Number of experiences to return when sampling.
        :type Batch_Size: int
        """
        self.gi_Batch_Size = Batch_Size
        self.gi_Memory_Capacity = Memory_Capacity
        self.Memory = deque(maxlen = self.gi_Memory_Capacity)

    def Remember(self, Date : tuple) -> None:
        """
        Add a new experience to the buffer.

        :param Date: Experience tuple (e.g., state, action, reward, next_state, done).
        :type Date: tuple
        """
        self.Memory.append(Date)

    def Batch(self) -> list:
        """
        Sample a batch of experiences from the buffer.

        :return: A list of sampled experiences.
        :rtype: list
        """
        return random.sample(self.Memory, min(self.gi_Batch_Size, len(self.Memory)))
    
    def Clear(self) -> None:
        """
        Clear all stored experiences in the buffer.
        """
        self.Memory.clear()

    def __len__(self) -> int:
        """
        Get the current number of stored experiences.

        :return: Number of experiences in the buffer.
        :rtype: int
        """
        return len(self.Memory)
    
    def __str__(self) -> str:
        """
        Get a string summary of the buffer parameters and status.

        :return: String representation of buffer status.
        :rtype: str
        """
        ls_Result = "Replay Buffer:\n"
        ls_Result += f"\tBatch Size: {self.gi_Batch_Size}\n"
        ls_Result += f"\tMemory Filled: {np.round(100 * len(self.Memory) / self.gi_Memory_Capacity, 2)}%\n"
        ls_Result += f"\tMemory Capacity: {self.gi_Memory_Capacity}\n"