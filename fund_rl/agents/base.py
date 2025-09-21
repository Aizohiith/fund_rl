from abc import ABC, abstractmethod
import numpy as np

class TAgent(ABC):
    """
    Base class for all reinforcement learning agents.
    """
    def __init__(self, Environment):
        self.Action_Space : int = Environment.action_space.n
        self.Reward : float = 0.0
        self.Loss : float = 0.0
        self.Environment_Name : str = Environment.spec.id if hasattr(Environment, 'spec') and Environment.spec else "Unknown Environment"
        ll_Space = Environment.observation_space

        if hasattr(ll_Space, 'shape') and ll_Space.shape != ():  # Box space or similar
            self.Observation_Space : int = ll_Space.shape[0]
        elif hasattr(ll_Space, 'n'):  # Discrete space
            self.Observation_Space : int = ll_Space.n
        else:
            raise ValueError("Unsupported observation space type")

        self.Is_Training : bool = True
        self.Name : str = "[Not Set]"
        self.Training_Steps : int = 0
        self.Evaluation_Steps : int = 0

    @abstractmethod
    def Choose_Action(self, State : np.ndarray) -> int:
        """
        Args:
            State: The current state.
        Returns:
            The chosen action.
        Choose an action based on the current state.
        """
        pass

    @abstractmethod
    def Update(self, Transition : tuple) -> None:
        """
        Args:
            Transition: The transition data to update the agent.
        Update the agent from experience.
        """
        pass

    @abstractmethod
    def Policy(self, State: np.ndarray) -> np.ndarray:
        """
        Get the action probabilities for the given state.
        Args:
            State: The current state.
        Returns:
            A list of action probabilities.
        """
        pass

    def Save(self, Path : str) -> None:
        """
        Save the agent's model to the specified path.
        Args:
            Path: Path to save the model.
        """
        raise NotImplementedError("Save method not implemented.")

    def Load(self, Path : str) -> None:
        """
        Load the agent's model from the specified path.
        Args:
            Path: Path to load the model from.
        """
        raise NotImplementedError("Load method not implemented.")
        
    def __str__(self) -> str:
        """ 
        String representation of the agent.
        Returns:
            str: String representation of the agent.
        """
        ls_Result =  f"\nAgent: {self.Name}\n"
        ls_Result += f"Environment:\n"
        ls_Result += f"\tName: {self.Environment_Name}\n"
        ls_Result += f"\tAction Space: {self.Action_Space}\n"
        ls_Result += f"\tObservation Space: {self.Observation_Space}\n"
        ls_Result += f"Training Mode: {self.Is_Training}\n"
        ls_Result += f"Training Steps: {self.Training_Steps}\n"
        ls_Result += f"Evaluation Steps: {self.Evaluation_Steps}\n"
        return ls_Result