from fund_rl.analyzers.base import TAnalyzer
from fund_rl.utility.filters import Mean_Filter, STD_Filter, EMA_Filter
from fund_rl.utility.format import Format
import matplotlib.pyplot as plt

class TPerformance_Analyzer(TAnalyzer):
    """
    Analyze the performance of the agent based on rewards and losses logged in the tracker.
    
    Args:
        Tracker (TMetric_Tracker, optional): An instance of TMetric_Tracker to log and retrieve metrics. Defaults to None.
    """
    def __init__(self, Tracker=None):
        super().__init__(Tracker)

    def Set_Tracker(self, Tracker):
        """
        Set the metric tracker for the analyzer and add required properties.
        Args:
            Tracker (TMetric_Tracker): An instance of TMetric_Tracker.
        """
        super().Set_Tracker(Tracker)

        self.gg_Tracker.Add_Property("Reward")
        self.gg_Tracker.Add_Property("Loss")

    def Analyze(self):
        """
        Analyze the logged rewards and losses, applying filters to smooth the data.
        Computes mean and standard deviation of rewards.
        """
        larr_Rewards = self.gg_Tracker.Data("Reward")
        larr_Losses = self.gg_Tracker.Data("Loss")
        larr_Mean_Rewards = Mean_Filter(larr_Rewards)
        larr_STD_Rewards = STD_Filter(larr_Rewards)

        self.gg_Report['Rewards'] = larr_Rewards
        self.gg_Report['Rewards Filtered'] = EMA_Filter(larr_Rewards, self.gg_Tracker.gf_Filter_Strength)
        self.gg_Report['Losses'] = larr_Losses
        self.gg_Report['Losses Filtered'] = EMA_Filter(larr_Losses, self.gg_Tracker.gf_Filter_Strength)
        self.gg_Report['Mean Rewards'] = larr_Mean_Rewards
        self.gg_Report['STD Rewards'] = larr_STD_Rewards
    
    def Plot(self):
        """
        Plot the rewards and losses along with their filtered versions.
        Creates a 2x2 subplot layout for better visualization.
        1. Agent's Rewards
        2. Agent's Losses
        3. Agent's Mean Rewards
        4. Agent's STD Rewards
        """
        plt.subplot(2, 2, 1)
        plt.plot(self.gg_Report['Rewards'], label='Rewards', color='blue', alpha=0.3)
        plt.plot(self.gg_Report['Rewards Filtered'], label='Filtered Rewards', color='blue')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title("Agent's Rewards")

        plt.subplot(2, 2, 2)
        plt.plot(self.gg_Report['Losses'], label='Losses', color='red', alpha=0.3)
        plt.plot(self.gg_Report['Losses Filtered'], label='Filtered Losses', color='red')
        plt.xlabel('Episodes')
        plt.ylabel('Losses')
        plt.title("Agent's Losses")

        plt.subplot(2, 2, 3)
        plt.plot(self.gg_Report['Mean Rewards'], label='Mean Rewards', color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.title("Agent's Mean Rewards")

        plt.subplot(2, 2, 4)
        plt.plot(self.gg_Report['STD Rewards'], label='STD Rewards', color='orange')
        plt.xlabel('Episodes')
        plt.ylabel('STD Rewards')
        plt.title("Agent's STD Rewards")

        plt.tight_layout()
        plt.show()

    
    def Print(self):
        """
        Print a summary report of the performance analysis.
        """
        larr_Report = ["Performance Analysis Report:"]
        larr_Report.append(f"Total Episodes: {len(self.gg_Report['Rewards'])}")
        larr_Report.append(f"Final Reward: {self.gg_Report['Rewards'][-1]:.2f}")
        larr_Report.append(f"Final Filtered Reward: {self.gg_Report['Rewards Filtered'][-1]:.2f}")
        larr_Report.append(f"Final Loss: {self.gg_Report['Losses'][-1]:.4f}")
        larr_Report.append(f"Final Filtered Loss: {self.gg_Report['Losses Filtered'][-1]:.4f}")
        larr_Report.append(f"Mean Final Reward: {self.gg_Report['Mean Rewards'][-1]:.2f}")
        larr_Report.append(f"STD of Final Reward: {self.gg_Report['STD Rewards'][-1]:.2f}")

        print(Format(larr_Report))
