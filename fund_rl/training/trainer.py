import gymnasium as gym
from fund_rl.trackers.base import TTracker
from fund_rl.agents.base import TAgent
from fund_rl.utility.run import Run_Agent

def Train_Agent(Environment: gym.Env, Agent: TAgent, Tracker: TTracker = None, Episodes: int = 200, Early_Stopping: float = None, Early_Stopping_Parameter = "Reward") -> None:
    """
    Trains an agent on the given Gym environment over a specified number of episodes.

    This function executes the training loop for the agent. Optionally, a tracker can be
    used to monitor the progress, and early stopping can be applied based on a performance metric.

    Parameters:
        Env (gym.Env): The Gym environment on which to train the agent.
        Agent (TAgent): The agent to be trained. Must have `Is_Training` set to True.
        Tracker (TTracker, optional): An optional tracker object for logging and monitoring training progress.
        Episodes (int): Number of training episodes. Defaults to 200.
        Early_Stopping (float, optional): If set, training will stop early if the specified
            parameter reaches or exceeds this threshold.
        Early_Stopping_Parameter (str): The name of the performance parameter to monitor for early stopping.
            Defaults to "Reward".

    Raises:
        ValueError: If the agent is not in training mode (`Is_Training` is False).

    Returns:
        None
    """
    # Check if the agent is in training mode
    if not Agent.Is_Training:
        raise ValueError("Agent is not in training mode. Set Is_Training to True before training.")

    # Initialize the tracker if provided
    if (Tracker is not None):
        Tracker.Set_Iterations(Episodes)
        Tracker.Start("Training: " + Agent.Name + " on " + Environment.spec.id)

    # Run the training loop
    Run_Agent(Environment, Agent, Tracker, Episodes, Early_Stopping, Early_Stopping_Parameter)

    # Finalize the tracker if provided
    if Tracker is not None:
        Tracker.Finish()
