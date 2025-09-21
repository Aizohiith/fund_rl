import numpy as np
from typing import Any

def Index_State(State : Any, Table : dict, Action_Space : int, Initial_Value : float = 0.0, Precision : int = 1) -> tuple:
    """
    This function takes a state representation and ensures it is in a consistent format for indexing.
    It converts the state into a tuple, rounding float values to a specified precision.
    If the state is not already in the provided table, it initializes an entry with a default value.
    Args:
        State (Any): The state representation, which can be a numpy array, int, or tuple.
        Table (dict): The dictionary to check and potentially add the state to.
        Action_Space (int): The size of the action space, used to initialize the state's entry in the table.
        Initial_Value (float): The initial value to assign if the state is not in the table.
        Precision (int): The number of decimal places to round float values to.
    Returns:
        tuple: The formatted state as a tuple.
    """
    ll_State_tuple = None

    if isinstance(State, (np.ndarray)):
        if isinstance(State[0], (float, np.float32, np.float64)):
            for C1 in range(len(State)):
                State[C1] = round(State[C1], Precision)
            ll_State_tuple = tuple(State.tolist())
        if isinstance(State[0], (int, np.uint8 ,np.integer, np.int32, np.int64)):
            ll_State_tuple = tuple(State.tolist())
    elif isinstance(State, (int, np.integer, np.int32, np.int64)):
        ll_State_tuple = (State,)
    elif isinstance(State, tuple):
        ll_State_tuple = State
    else:
        ll_State_tuple = tuple(map(tuple, State))  # Convert to a tuple

    if not ll_State_tuple in Table:
        Table[ll_State_tuple] = np.full(Action_Space, Initial_Value)
    return ll_State_tuple