from fund_rl.analyzers.base import TAnalyzer
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym
import numpy as np
from fund_rl.agents.base import TAgent
from fund_rl.training.trainer import Train_Agent
from fund_rl.trackers.metric import TMetric_Tracker
import matplotlib.pyplot as plt
from fund_rl.utility.format import Format

class TConsistency_Analyzer(TAnalyzer):
    """
    Compares the performance consistency of multiple instances of a given agent class
    on a specified environment over several training iterations.
    Args:
        Environment (gym.Env): The environment in which the agent will be trained.
        Agent_Class (Type[TAgent]): The class of the agent to be instantiated and trained.
        Config (dict): Configuration parameters to be passed to the agent constructor.
        Episodes (int): Number of episodes to train each agent instance.
        Iterations (int): Number of independent training iterations to perform.
        Workers (int): Number of parallel workers to use for training.
        Tracker (TMetric_Tracker, optional): An optional metric tracker for logging training metrics.
    """
    def __init__(self, Environment, Agent_Class : TAgent, Config: dict, Episodes: int = 200, Iterations: int = 100, Workers: int = 4, Tracker=None):
        super().__init__(Tracker)

        self.gg_Agent_Class = Agent_Class
        self.gg_Environment = Environment
        self.gg_Config = Config
        self.gi_Episodes = Episodes
        self.gi_Iterations = Iterations
        self.gi_Workers = Workers

    def Analyze(self):
        """
        Conducts the consistency analysis by training multiple instances of the agent class
        on the environment for a specified number of episodes and iterations. The results are collected and stored in the analysis report.
        """
        larr_Arguments = [(self.gg_Environment, self.gg_Agent_Class, self.gg_Config, self.gi_Episodes) for _ in range(self.gi_Iterations)]

        self.gg_Tracker.Set_Iterations(self.gi_Iterations)
        self.gg_Tracker.garr_Log_Data.clear()
        self.gg_Tracker.Add_Property("Agent")
        self.gg_Tracker.Add_Property("Rewards")
        self.gg_Tracker.Start("Consistency Test: " + self.gg_Agent_Class.__name__ + " on " + self.gg_Environment.spec.id)

        C1 = 0
        with ProcessPoolExecutor(max_workers=self.gi_Workers) as Executor:
            for larr_Result, ll_Agent in Executor.map(Training, larr_Arguments):
                self.gg_Tracker.Log("Rewards", larr_Result, C1)
                self.gg_Tracker.Log("Agent", ll_Agent, C1)
                C1 += 1

        larr_Results = np.array(self.gg_Tracker.Data("Rewards"))
        larr_Mean_Results = larr_Results.mean(axis=0)
        larr_STD_Results = larr_Results.std(axis=0)

        self.gg_Report['Mean Rewards'] = larr_Mean_Results
        self.gg_Report['STD Rewards'] = larr_STD_Results
        self.gg_Report['All Rewards'] = larr_Results
        self.gg_Report['Agents'] = self.gg_Tracker.Data("Agent")

        self.gg_Tracker.Finish()

    def Plot(self, Save_Path=None):
        """
        Plots the mean rewards and standard deviation over episodes from the consistency analysis.
        """
        plt.plot(self.gg_Report['Mean Rewards'], label='Mean Rewards')
        plt.fill_between(range(len(self.gg_Report['Mean Rewards'])), self.gg_Report['Mean Rewards'] - self.gg_Report['STD Rewards'], self.gg_Report['Mean Rewards'] + self.gg_Report['STD Rewards'], alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Agent Training Rewards')
        if Save_Path is not None:
            plt.savefig(Save_Path)
        else:
            plt.show()

    def Print(self, Save_Path=None):
        """
        Prints a summary report of the consistency analysis results.
        """
        larr_Report = ["Consistency Analysis Report:"]
        larr_Report.append(f"Number of Iterations: {self.gi_Iterations}")
        larr_Report.append(f"Number of Episodes per Iteration: {self.gi_Episodes}")
        larr_Report.append(f"Mean Final Reward: {self.gg_Report['Mean Rewards'][-1]:.2f}")
        larr_Report.append(f"STD of Final Reward: {self.gg_Report['STD Rewards'][-1]:.2f}")
        if Save_Path is not None:
            with open(Save_Path, 'w') as f:
                f.write(Format(larr_Report))
        else:
            print(Format(larr_Report))

def Training(pp_Arguments : tuple) -> tuple:
    """
    Trains a single agent instance on the given environment for a fixed number of episodes.

    This function initializes a metric tracker, constructs the agent with the provided configuration,
    trains the agent using the `Train_Agent` function, and returns both the reward history and
    the trained agent.

    Parameters:
        pp_Arguments (tuple): A tuple containing:
            - gym.Env: The environment to train the agent in.
            - Type[TAgent]: The agent class to instantiate.
            - dict: Configuration dictionary passed to the agent constructor.
            - int: Number of episodes to train the agent for.

    Returns:
        tuple:
            - np.ndarray: Array of rewards collected over the training episodes.
            - TAgent: The trained agent instance.
    """
    ll_Env, ll_Agent_Class, ll_Config, li_Episodes = pp_Arguments
    ll_Tracer = TMetric_Tracker(Show_Progress=False)
    ll_Agent = ll_Agent_Class(ll_Env, **ll_Config)
    Train_Agent(ll_Env, ll_Agent, ll_Tracer, li_Episodes)
    return ll_Tracer.Data("Reward"), ll_Agent