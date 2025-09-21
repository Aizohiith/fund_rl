import matplotlib.pyplot as plt
from fund_rl.utility.filters import EMA
from fund_rl.trackers.base import TTracker

class TLocal_Tracker(TTracker):
    """
    A local tracking class that extends TTracker to log, store, and visualize
    training metrics locally using matplotlib.
    """
    def __init__(self, Log_Data : list = None, Show_Progress : bool = True, Filter_Strength: float = 0.99) -> None:
        """
        Initializes the TLocal_Tracker object.

        Args:
            Log_Data (list): Initial log data, defaults to None.
            Show_Progress (bool): Whether to print logging progress, defaults to True.
            Filter_Strength (float): Smoothing factor for EMA filtering, defaults to 0.99.
        """
        super().__init__(Log_Data, Show_Progress=Show_Progress, Filter_Strength=Filter_Strength)
        self.garr_Session_Data = []

    def Start(self, Name: str) -> None:
        """
        Starts a new tracking session.

        Args:
            Name (str): Name of the session.
        """
        super().Start(Name)
        self.garr_Session_Data = []

    def Log(self, Key: str, Value: float, Step: int) -> None:
        """
        Logs a metric value at a specific step.

        Args:
            Key (str): The name of the metric.
            Value (float): The value of the metric.
            Step (int): The step index at which the value is logged.
        """
        super().Log(Key, Value, Step)

        if ("Filtered_" + Key in self.garr_Log_Data):
            self.garr_Session_Data.append({"Step": Step, "Metric": "Filtered_" + Key, "Value": self.gg_Filtered_Data[Key]})

        self.garr_Session_Data.append({"Step": Step, "Metric": Key, "Value": Value})

    def Finish(self) -> None:
        """
        Ends the current session and plots logged data using matplotlib.
        Plots both raw and filtered metric values for each tracked key.
        """
        super().Finish()

        # Plotting the rewards data
        for ls_Key in self.garr_Log_Data:
            if ls_Key.startswith("Filtered_"):
                continue
            larr_Data = [entry["Value"] for entry in sorted(self.garr_Session_Data, key=lambda x: x['Step']) if entry["Metric"] == ls_Key]
            larr_Filtered_Data = []
            for C1 in range(len(larr_Data)):
                if (C1 == 0):
                    larr_Filtered_Data.append(larr_Data[C1])
                else:
                    larr_Filtered_Data.append(EMA(larr_Filtered_Data[-1], larr_Data[C1], Alpha=self.gf_Filter_Strength))
            plt.plot(larr_Data, alpha=0.25, label="Original Data", color="blue")
            plt.plot(larr_Filtered_Data, label="Filtered Data", color="blue")
            plt.title(ls_Key)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

