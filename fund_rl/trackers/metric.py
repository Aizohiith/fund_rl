import json
import os
from fund_rl.trackers.base import TTracker

class TMetric_Tracker(TTracker):
    """
    A local metric tracker for recording, saving, and loading session metrics.

    This class extends TTracker and allows for structured metric logging
    that can be filtered, persisted to disk, and reloaded for analysis.
    """
    def __init__(self, Log_Data : list = None, Show_Progress : bool = True, Filter_Strength: float = 0.99) -> None:
        """
        Initializes the TMetric_Tracker object.

        Args:
            Log_Data (list): Initial log data, defaults to None.
            Show_Progress (bool): Whether to print logging progress, defaults to True.
            Filter_Strength (float): Smoothing factor for EMA filtering, defaults to 0.99.
        """
        super().__init__(Log_Data, Show_Progress=Show_Progress, Filter_Strength=Filter_Strength)
        self.garr_Session_Data = []

    def Start(self, Name: str) -> None:
        """
        Start a new tracking session.

        Args:
            Name (str): The name of the session.
        """
        super().Start(Name)
        self.garr_Session_Data = []

    def Log(self, Key: str, Value: float, Step: int) -> None:
        """
        Log a new value for a given metric at a specific step.

        Args:
            Key (str): The name of the metric.
            Value (float): The value to log.
            Step (int): The training step associated with the value.
        """
        super().Log(Key, Value, Step)

        if ("Filtered_" + Key in self.garr_Log_Data):
            self.garr_Session_Data.append({"Step": Step, "Metric": "Filtered_" + Key, "Value": self.gg_Filtered_Data[Key]})

        self.garr_Session_Data.append({"Step": Step, "Metric": Key, "Value": Value})
    
    def Data(self, Key: str, Sort_Key: str = "Step") -> list:
        """
        Retrieve a list of values for a given metric, sorted by a key.

        Args:
            Key (str): The metric to extract.
            Sort_Key (str): The key to sort the data by (default is "Step").

        Returns:
            list: A list of values corresponding to the metric.
        """
        return [entry["Value"] for entry in sorted(self.garr_Session_Data, key=lambda x: x[Sort_Key]) if entry["Metric"] == Key]

    def Save(self, File: str) -> None:
        """
        Save the session data to a JSON file.

        Args:
            File (str): The file path where data should be saved.
        """
        with open(File, 'w') as f:
            json.dump(self.garr_Session_Data, f, indent=4)

    def Load(self, File: str) -> None:
        """
        Load session data from a JSON file.

        Args:
            File (str): The file path to load data from.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(File):
            raise FileNotFoundError(f"File {File} does not exist.")

        with open(File, 'r') as f:
            self.garr_Session_Data = json.load(f)

    def Finish(self) -> None:
        """
        Finalize the tracking session.
        """
        super().Finish()

