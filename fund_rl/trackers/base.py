from tqdm import tqdm
import numbers
from fund_rl.utility.format import Format
from fund_rl.utility.filters import EMA
from typing import Any

class TTracker():
    """
    A utility class for tracking and logging training metrics with optional exponential smoothing
    and tqdm progress bar display.
    """
    def __init__(self, Log_Data : list = None, Show_Progress : bool = True, Filter_Strength : float = 0.99) -> None:
        """
        Initialize the tracker.

        Args:
            Log_Data (list[str]): List of properties to track. Defaults to ["Reward"].
            Show_Progress (bool): Whether to display a tqdm progress bar. Defaults to True.
            Filter_Strength (float): EMA smoothing factor (0 = no smoothing, 1 = infinite smoothing). Defaults to 0.99.
        """
        if (Log_Data is None):
            Log_Data = ["Reward"]
        self.garr_Log_Data = list(Log_Data)
        self.gi_Iterations = 0
        self.gb_Show_Progress = Show_Progress
        self.gf_Filter_Strength= Filter_Strength
        self.gg_Filtered_Data = {}
        self.gg_Progress_Bar = None
        self.gs_Name = "[not set]"

    def Set_Iterations(self, Iterations: int) -> None:
        """
        Set the number of iterations for the tracker.

        Args:
            Iterations (int): Number of iterations to track.
        """
        self.gi_Iterations = Iterations

    def Log(self, Key: str, Value: Any, Step: int) -> None:
        """
        Log a value for a tracked property.

        Args:
            Key (str): Property name.
            Value (Any): Value to log.
            Step (int): Current step or iteration.
        """
        if isinstance(Value, numbers.Number):
            if (self.gg_Filtered_Data.get(Key) is None):
                self.gg_Filtered_Data[Key] = Value
            else:
                self.gg_Filtered_Data[Key] = EMA(self.gg_Filtered_Data[Key], Value, self.gf_Filter_Strength)

        if (not self.gb_Show_Progress):
            return
        if self.gg_Progress_Bar is None:
            return

        self.gg_Progress_Bar.n = Step

        ll_Log = {}

        for E1 in self.gg_Filtered_Data.keys():
                ll_Log[E1[0]] = f"{self.gg_Filtered_Data[E1]:.2f}"

        self.gg_Progress_Bar.set_postfix(ll_Log)
        self.gg_Progress_Bar.update(0)

    def Start(self, Name: str) -> None:
        """
        Start a new tracking session.

        Args:
            Name (str): The name of the session.
        """
        self.gs_Name = Name
        self.gg_Filtered_Data = {}
        if (not self.gb_Show_Progress):
            return

        ls_Name = f"{Name}"
        ls_Properties = f"Tracking properties:"

        larr_Print = [ls_Name, ls_Properties]
        for ls_Property in self.garr_Log_Data:
            larr_Print.append(f" - {ls_Property}")

        print(Format(larr_Print))

        self.gg_Progress_Bar = tqdm(total=self.gi_Iterations, desc='')

    def Finish(self) -> None:
        """
        Finish the current tracking session and print final results.
        """
        if (not self.gb_Show_Progress):
            return
        if self.gg_Progress_Bar is None:
            return

        self.gg_Progress_Bar.n = self.gi_Iterations
        self.gg_Progress_Bar.close()

        ls_Name = f"{self.gs_Name}"
        larr_Print = [ls_Name, "Final properties:"]
        for ls_Property in self.garr_Log_Data:
            ll_Value = self.gg_Filtered_Data.get(ls_Property)
            if isinstance(ll_Value, numbers.Number):
                larr_Print.append(f" - {ls_Property}: {ll_Value:.2f}")
            else:
                larr_Print.append(f" - {ls_Property}: N/A")

        print(Format(larr_Print))

    def Add_Property(self, Name: str) -> None:
        """
        Add a property to be tracked.

        Args:
            Name (str): Name of the new property.
        """
        if (Name not in self.garr_Log_Data):
            self.garr_Log_Data.append(Name)
