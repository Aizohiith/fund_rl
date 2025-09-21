from fund_rl.action_selection.base import TAction_Selection
from fund_rl.utility.index_state import Index_State
import numpy as np
from typing import Any

class TUCB_Action_Selection(TAction_Selection):
    """
    Upper Confidence Bound (UCB) action selection strategy.

    This class implements UCB exploration for discrete action spaces. It maintains a visitation count
    for each state-action pair and computes exploration bonuses based on visit frequency.

    Args:
        Action_Space (int): Number of discrete actions available.
        C (float): Exploration constant controlling trade-off between exploration and exploitation. Defaults to 2.
        Precision (int): Number of decimal places to round the state representation for hashing. Defaults to 1.
    """
    def __init__(self, Action_Space : int, C : float = 2, Precision : int = 1) -> None:
        self.C = C
        self.Precision = Precision
        self.Action_Space = Action_Space
        self.N_Table = {}

    def Take_Action(self, State : Any, Action : int) -> None:
        """
        Record the selection of an action for a given state.

        Args:
            State (Any): The current environment state. Must be indexable with `Index_State`.
            Action (int): The action taken.
        """
        ll_State = Index_State(State, self.N_Table, self.Action_Space, 1, self.Precision)
        self.N_Table[ll_State][Action] += 1

    def Exploration_Values(self, State : Any, Training_Steps : int) -> np.ndarray:
        """
        Compute the UCB exploration value for a given state.

        Args:
            State (Any): The current environment state.
            Training_Steps (int): Total number of training steps so far.

        Returns:
            array: The computed exploration bonus for the state.
        """
        ll_State = Index_State(State, self.N_Table, self.Action_Space, 1, self.Precision)
        return self.C * np.sqrt((np.log(Training_Steps+1))/(self.N_Table[ll_State]))

    def __str__(self) -> str:
        """
        Return a string representation of the UCB configuration.

        Returns:
            str: A summary of the UCB settings and current state count.
        """
        ls_Result = "UCB:\n"
        ls_Result += f"\tC: {self.C}\n"
        ls_Result += f"\tPrecision: {self.Precision}\n"
        ls_Result += f"\tStates: {len(self.N_Table)}\n"
        return ls_Result
