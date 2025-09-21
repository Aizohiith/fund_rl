from fund_rl.action_selection.base import TAction_Selection
import numpy as np

class TEpsilon_Greedy_Action_Selection(TAction_Selection):
    """
    Epsilon-Greedy action selection strategy.

    This strategy selects a random action with probability ε (exploration),
    and the best-known action with probability 1 - ε (exploitation). The
    exploration rate ε decays over time based on a multiplicative decay factor.

    Args:
        Exploration_Rate (float): Current exploration probability ε.
        Exploration_Decay (float): Multiplicative decay factor for ε.
        Min_Exploration_Rate (float): Lower bound for ε to maintain some exploration.
    """
    def __init__(self, Exploration_Rate : float = 1.0, Exploration_Decay : float = 0.995, Min_Exploration_Rate : float = 0.01) -> None:
        self.Exploration_Rate = Exploration_Rate
        self.Exploration_Decay = Exploration_Decay
        self.Min_Exploration_Rate = Min_Exploration_Rate

    def Decay_Exploration_Rate(self) -> None:
        """
        Decay the current exploration rate ε by the decay factor.

        Ensures ε remains above the minimum exploration rate.
        """
        if (self.Exploration_Rate > self.Min_Exploration_Rate):
            self.Exploration_Rate = np.max([self.Min_Exploration_Rate, self.Exploration_Rate * self.Exploration_Decay])

    def Calculate_Decay_Factor(self, Steps : int, Reach : float = 0.7, Final : float = 0.01) -> None:
        """
        Calculate and set the decay factor such that ε reaches `Final` after
        `Steps * Reach` steps.

        Args:
            Steps (int): Total number of expected training steps.
            Reach (float): Proportion of steps after which ε should reach its target.
            Final (float): Target final value for ε.
        """
        self.Exploration_Decay = np.pow(Final, 1 / (Steps * Reach))

    def Should_Explore(self) -> bool:
        """
        Decide whether to explore or exploit.

        Returns:
            bool: True if the agent should explore (random action),
                  False if it should exploit (greedy action).
        """
        return np.random.rand() < self.Exploration_Rate

    def __str__(self) -> str:
        """
        Return a string summary of the epsilon-greedy parameters.

        Returns:
            str: A formatted string showing current epsilon parameters.
        """
        ls_Result = "ε Greedy:\n"
        ls_Result += "\tExploration Rate: " + str(self.Exploration_Rate) + "\n"
        ls_Result += "\tExploration Decay: " + str(self.Exploration_Decay) + "\n"
        ls_Result += "\tMin Exploration Rate: " + str(self.Min_Exploration_Rate) + "\n"
        return ls_Result
