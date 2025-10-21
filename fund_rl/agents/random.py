from fund_rl.agents.base import TAgent


class TRandom_Agent(TAgent):
    """
    A random agent that selects actions uniformly at random.
    Args:
        Environment (gym.Env): The environment in which the agent operates.
    """
    def __init__(self, Environment):
        super().__init__(Environment=Environment)
        self.gg_Environment = Environment
        self.Name = "Random"

    def Choose_Action(self, State):
        """
        Chooses an action randomly from the action space.
        Args:
            State: The current state (not used in random agent).
        """

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
            State: The current state (not used in random agent).
        """
        lf_Probability = 1.0 / self.Action_Space
        return [lf_Probability] * self.Action_Space