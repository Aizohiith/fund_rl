import torch
import torch.nn as nn

class TNetwork(nn.Module):
    """
    Base class for neural network architectures used in reinforcement learning agents.

    Attributes:
        Input_Dimension (int): Dimension of the input features.
        Output_Dimension (int): Dimension of the output features (e.g., number of actions or value predictions).
    """
    def __init__(self, Input_Dimensions : int, Output_Dimensions : int) -> None:
        super().__init__()
        self.gi_Input_Dimensions = Input_Dimensions
        self.gi_Output_Dimensions = Output_Dimensions

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        This method must be overridden by subclasses.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def Save(self, File_Name : str) -> None:
        """
        Saves the model's parameters to a file.

        Args:
            File_Name (str): The base name of the file ('.pth' extension will be appended automatically).
        """
        torch.save(self.state_dict(), File_Name + ".pth")

    def Load(self, File_Name : str) -> None:
        """
        Loads the model's parameters from a file.

        Args:
            File_Name (str): The base name of the file ('.pth' extension will be appended automatically).
        """
        self.load_state_dict(torch.load(File_Name + ".pth"))
        self.eval()