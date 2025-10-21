from fund_rl.agents.base import TAgent
import numpy as np


class TRandom_Agent(TAgent):
    """
    A random agent that selects actions uniformly at random.
    Args:
        Environment (gym.Env): The environment in which the agent operates.
        Action_Mask_Function (callable, optional): A function that takes the current state and returns a mask of valid actions.
    """
    def __init__(self, Environment, Action_Mask_Function=None):
        super().__init__(Environment=Environment)
        self.gg_Environment = Environment
        self.Name = "Random"
        self.Action_Mask_Function = Action_Mask_Function

    def Choose_Action(self, State):
        """
        Chooses an action randomly from the action space.
        Args:
            State: The current state.
        """
        if self.Action_Mask_Function:
            larr_Action_Mask = self.Action_Mask_Function(State)
            larr_Valid_Actions = [C1 for C1, lb_Valid in enumerate(larr_Action_Mask) if lb_Valid]
            return np.random.choice(larr_Valid_Actions)
        return self.gg_Environment.action_space.sample()

    def Update(self, Transition):
        """
        Updates the agent's knowledge based on the transition.
        Args:
            Transition: The transition data (not used in random agent).
        """
        pass

    def Policy(self, State):
        """
        Get the action probabilities for the given state.
        Args:
            State: The current state.
        """
        if self.Action_Mask_Function:
            larr_Action_Mask = self.Action_Mask_Function(State)
            larr_Valid_Actions = [C1 for C1, lb_Valid in enumerate(larr_Action_Mask) if lb_Valid]
            lf_Probability = 1.0 / len(larr_Valid_Actions)
            larr_Probabilities = [lf_Probability if C1 in larr_Valid_Actions else 0.0 for C1 in range(self.Action_Space)]
            return larr_Probabilities
        
        lf_Probability = 1.0 / self.Action_Space
        return [lf_Probability] * self.Action_Space
    
    def __str__(self):
        """
        String representation of the agent.
        """
        ls_Result =  super().__str__()
        ls_Result += f"Action Mask Function: {self.Action_Mask_Function.__name__}" if self.Action_Mask_Function else "No Action Mask Function\n"
        return ls_Result