class TAction_Selection():
    """
    Base class for action selection strategies used in reinforcement learning agents.

    This class defines the interface for selecting actions from a given policy or value function.
    Derived classes should implement specific strategies such as epsilon-greedy, softmax, or UCB.

    Intended to be subclassed and extended with concrete implementations.
    """
