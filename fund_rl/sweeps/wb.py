import wandb
import gymnasium as gym
from fund_rl.agents.base import TAgent
from fund_rl.training.trainer import Train_Agent
from fund_rl.trackers.wb import TWandB_Tracker

def WandB_Sweep(Environment: gym.Env, Agent_Class: TAgent, Project_Name: str, Entity: str, Config: dict, Episodes: int = 200) -> None:
    """
    Run a Weights & Biases (wandb) hyperparameter sweep for a reinforcement learning agent.

    This function initializes a wandb sweep using the provided configuration, trains the agent
    using the environment and logs data to wandb using TWandB_Tracker.

    Args:
        Environment (gym.Env): The Gymnasium environment for training the agent.
        Agent_Class (TAgent): The class of the RL agent to be trained.
        Project_Name (str): The name of the wandb project.
        Entity (str): The wandb entity (user or team).
        Config (dict): The configuration dictionary for the wandb sweep (includes search space).
        Episodes (int): The number of training episodes per agent run.
    """

    def Train():
        """
        Train function used by wandb.agent to execute one sweep trial.
        Initializes the tracker, agent, and starts the training loop.
        """
        wandb.init()
        ll_Config = wandb.config

        ll_Tracker = TWandB_Tracker(Project_Name=Project_Name, Entity=Entity, Show_Progress=False)
        ll_Tracker.Add_Property("Filtered_Reward")
        ll_Agent = Agent_Class(Environment, **ll_Config)

        Train_Agent(Environment, ll_Agent, ll_Tracker, Episodes)

    li_Sweep_id = wandb.sweep(Config, project=Project_Name, entity=Entity)
    wandb.agent(li_Sweep_id, function=Train)