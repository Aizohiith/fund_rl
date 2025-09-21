import gymnasium as gym
from fund_rl.trackers.base import TTracker
from fund_rl.agents.base import TAgent
from fund_rl.utility.run import Run_Agent

def Evaluate_Agent(Environment: gym.Env, Agent: TAgent, Tracker: TTracker = None, Episodes: int = 200) -> None:
    """
    Evaluates a trained agent on the given Gym environment over a number of episodes.

    This function runs the agent in evaluation mode for a specified number of episodes
    and optionally tracks performance using the provided tracker.

    Parameters:
        Environment (gym.Env): The Gym environment to evaluate the agent on.
        Agent (TAgent): The agent to be evaluated. Must have `Is_Training` set to False.
        Tracker (TTracker, optional): An optional tracker object for logging and tracking the evaluation progress.
        Episodes (int): The number of episodes to evaluate the agent for. Default is 200.

    Raises:
        ValueError: If the agent is still in training mode (`Is_Training` is True).

    Returns:
        None
    """
    if Agent.Is_Training:
        raise ValueError("Agent is in training mode. Set Is_Training to False before evaluation.")

    if (Tracker is not None):
        Tracker.Set_Iterations(Episodes)
        Tracker.Start("Evaluating: " + Agent.Name + " on " + Environment.spec.id)

    Run_Agent(Environment, Agent, Tracker, Episodes)

    if Tracker is not None:
        Tracker.Finish()