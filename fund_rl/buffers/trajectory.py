from fund_rl.buffers.base import TBuffer
from typing import Any

class TTrajectory_Buffer(TBuffer):
    """
    Buffer for storing full trajectories (episodes) of experience.

    Useful for on-policy RL methods that operate over complete episodes.

    Inherits from :class:`TBuffer`.
    """
    def __init__(self) -> None:
        """
        Initialize an empty trajectory buffer to store sequential experience steps.
        """
        self.Memory = []
 
    def Remember(self, pp_Date : tuple) -> None:
        """
        Store a single step in the trajectory buffer.

        :param pp_Date: A tuple representing one step e.g. -> (state, action, reward, next_state, done).
        :type pp_Date: tuple
        """
        self.Memory.append(pp_Date)

    def __len__(self) -> int:
        """
        Return the number of stored steps in the current trajectory.

        :return: Number of elements in the buffer.
        :rtype: int
        """
        return len(self.Memory)
    
    def Clear(self) -> None:
        """
        Clear the stored trajectory.
        """
        self.Memory = []
    
    def Trajectory(self, pi_Index : int = -1) -> Any:
        """
        Return the full trajectory or a specific step by index.

        :param pi_Index: Step index to return; -1 returns the full trajectory.
        :type pi_Index: int
        :return: List of experiences or a single experience.
        :rtype: Any
        """
        if (pi_Index == -1):
            return self.Memory
        else:
            return self.Memory[pi_Index]
    
    def __str__(self) -> str:
        """
        Return a string summary of the trajectory buffer.

        :return: Buffer details.
        :rtype: str
        """
        ls_Result = "Trajectory Buffer\n"
        ls_Result += f"\tMemory Size: {len(self.Memory)}\n"
        return ls_Result