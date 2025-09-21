import wandb
from fund_rl.trackers.base import TTracker
from tqdm import tqdm

class TWandB_Tracker(TTracker):
    """
    Tracker that integrates with Weights & Biases (wandb) for experiment tracking.

    This tracker logs metrics to a wandb run, supporting filtered and raw data logging,
    and manages the lifecycle of the wandb run.
    """
    def __init__(self, Project_Name: str, Entity: str, Show_Progress : bool = True, Log_Data: list = None, Filter_Strength : float = 0.99) -> None:
        """
        Initialize the wandb tracker.

        Args:
            Project_Name (str): The wandb project name.
            Entity (str): The wandb entity (user or team).
            Show_Progress (bool): Whether to show progress bar updates.
            Log_Data (list): Initial log data.
            Filter_Strength (float): Filter strength for metric smoothing.
        """
        super().__init__(Log_Data=Log_Data, Show_Progress=Show_Progress, Filter_Strength=Filter_Strength)
        self.Project_Name = Project_Name
        self.Entity = Entity
        self.gg_Run = None

    def Start(self, Name: str) -> None:
        """
        Start a new wandb run.

        Args:
            Name (str): Name of the run.
        """
        super().Start(Name)
        self.gg_Run = wandb.init(project=self.Project_Name, entity=self.Entity, name=Name)

    def Log(self, Key: str, Value: float, Step: int) -> None:
        """
        Log a metric to wandb.

        Args:
            Key (str): The metric key.
            Value (float): The metric value.
            Step (int): The training step number.
        """
        super().Log(Key, Value, Step)

        if self.gg_Run is None:
            return

        if ("Filtered_" + Key in self.garr_Log_Data):
            wandb.log({"Filtered_" + Key: self.gg_Filtered_Data[Key]}, step=Step)

        wandb.log({Key: Value}, step=Step)

    def Finish(self) -> None:
        """
        Finish the wandb run and cleanup.
        """
        super().Finish()

        if self.gg_Run is None:
            return
        
        wandb.finish()