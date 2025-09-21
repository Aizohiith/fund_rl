import gymnasium as gym
from fund_rl.trackers.base import TTracker
from fund_rl.agents.base import TAgent
from fund_rl.utility.format import Format
import numpy as np


def Run_Agent(Environment: gym.Env, Agent: TAgent, Tracker: TTracker = None, Episodes: int = 200, Early_Stopping: float = None, Early_Stopping_Parameter = "Reward") -> None:
    """
    Runs the agent in the specified environment for a given number of episodes.

    The function executes the main loop for agent-environment interaction. It tracks
    performance using an optional tracker and supports early stopping if a specified
    performance metric exceeds a given threshold.

    Parameters:
        Environment (gym.Env): The Gymnasium environment in which the agent operates.
        Agent (TAgent): The agent to run. Must implement `Choose_Action` and `Update`.
        Tracker (TTracker, optional): Optional tracker for logging and monitoring performance.
        Episodes (int): The number of episodes to run the agent. Defaults to 200.
        Early_Stopping (float, optional): Threshold value for early stopping. If set, training stops
            once the specified tracker parameter reaches or exceeds this value.
        Early_Stopping_Parameter (str): The name of the parameter in the tracker to monitor for early stopping.
            Must be present in `Tracker.Log_Data` if `Early_Stopping` is used.

    Raises:
        TypeError: If input types do not match expected types.
        ValueError: If early stopping is requested but the tracker is missing or the parameter is not logged.

    Returns:
        None
    """
    
    # Input validation
    if isinstance(Environment, gym.Env) is False:
        raise TypeError("Environment must be an instance of gym.Env.")
    if isinstance(Agent, TAgent) is False:
        raise TypeError("Agent must be an instance of TAgent.")
    if Tracker is not None and isinstance(Tracker, TTracker) is False:
        raise TypeError("Tracker must be an instance of TTracker or None.")
    if Early_Stopping is not None and not isinstance(Early_Stopping, (float, int)):
        raise TypeError("Early_Stopping must be a float or int.")
    if Early_Stopping is not None and Tracker is None:
        raise ValueError("Early stopping requires a tracker to monitor the agent's performance.")
    if Early_Stopping is not None and Tracker is not None:
        if not Early_Stopping_Parameter in Tracker.garr_Log_Data:
            raise ValueError(f"Early stopping requires '{Early_Stopping_Parameter}' to be logged in the tracker.")

    # Training loop
    for li_Episode in range(Episodes):
        # Reset the environment at the start of each episode
        ll_State, _ = Environment.reset()
        # Initialize variables to track rewards and steps
        lf_Total_Reward = 0
        li_Step = 0

        # Episode loop
        while True:
            # Render the environment if applicable
            if (Environment.render_mode == "human" or Environment.render_mode == "rgb_array"):
                Environment.render()

            # Agent selects action based on current state
            li_Action = Agent.Choose_Action(ll_State)
            # Take action in the environment
            ll_Next_State, lf_Reward, pb_Done, pb_Truncated, _ = Environment.step(li_Action)
            # Update the agent with the transition
            li_Step += 1
            pb_Done = pb_Done or pb_Truncated
            Agent.Update((ll_State, li_Action, lf_Reward, ll_Next_State, pb_Done))
            if (Agent.Is_Training):
                Agent.Training_Steps += 1
            else:
                Agent.Evaluation_Steps += 1

            # Log data if a tracker is provided
            if Tracker is not None:
                if ("State" in Tracker.garr_Log_Data):
                    Tracker.Log("State", ll_State, li_Episode)
                if ("Action" in Tracker.garr_Log_Data):
                    Tracker.Log("Action", li_Action, li_Episode)
                if ("Done" in Tracker.garr_Log_Data):
                    Tracker.Log("Done", pb_Done, li_Episode)
                if ("Next_State" in Tracker.garr_Log_Data):
                    Tracker.Log("Next_State", ll_Next_State, li_Episode)

            # Update total reward
            lf_Total_Reward += lf_Reward

            # Update the state
            ll_State = ll_Next_State

            # Check if the episode is done
            if pb_Done:
                break
        
        # Log additional episode data
        if Tracker is not None:
            if ("Step" in Tracker.garr_Log_Data):
                Tracker.Log("Step", li_Step, li_Episode)

        # Update agent's reward attribute
        Agent.Reward = lf_Total_Reward

        # Log Agent attributes
        if Tracker is not None:
            for T1 in Tracker.garr_Log_Data:
                if hasattr(Agent, T1):
                    Tracker.Log(T1, getattr(Agent, T1), li_Episode)

            # Check for early stopping
            if Early_Stopping is not None:
                if np.round(Tracker.gg_Filtered_Data[Early_Stopping_Parameter], 2) >= Early_Stopping:
                    print("\n" + Format([f"Early stopping at episode {li_Episode + 1} with {Early_Stopping_Parameter} {np.round(Tracker.gg_Filtered_Data[Early_Stopping_Parameter], 2)}"]))
                    break