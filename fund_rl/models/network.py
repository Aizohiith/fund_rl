from fund_rl.models.base import TNetwork
import torch
import torch.nn as nn
from typing import Optional, Callable, Type

class TBasic_Network(TNetwork):
    """
    A configurable multi-layer perceptron (MLP) network with optional activation, output activation, 
    and layer initialization.

    Attributes:
        Input_Dimension (int): Dimension of the input features.
        Output_Dimension (int): Dimension of the output features.
        Hidden_Dimensions (int): Size of hidden layers.
        Hidden_Layers (int): Number of hidden layers.
        Final_Activation (Optional[Callable[[], nn.Module]]): Optional final activation (e.g., nn.Softmax).
        Activation (Type[nn.Module]): Activation function to use in hidden layers.
        Layer_Initialization (Optional[Callable[[nn.Module], nn.Module]]): Function to initialize layers.
    """
    def __init__(self, Input_Dimension : int, Output_Dimension : int, Hidden_Dimensions : int = 128, 
                 Hidden_Layers : int = 1, Final_Activation : Optional[Callable[[], nn.Module]] = None, Activation : Type[nn.Module] = nn.Tanh,
                 Layer_Initialization : Optional[Callable[[nn.Module], nn.Module]] = None) -> None:
        super().__init__(Input_Dimension, Output_Dimension)

        larr_Layers = []

        # Input layer
        if Layer_Initialization is not None:
            larr_Layers.append(Layer_Initialization(nn.Linear(Input_Dimension, Hidden_Dimensions)))
        else:
            larr_Layers.append(nn.Linear(Input_Dimension, Hidden_Dimensions))
        larr_Layers.append(Activation())

        # Hidden layers
        for C1 in range(Hidden_Layers):
            if Layer_Initialization is not None:
                larr_Layers.append(Layer_Initialization(nn.Linear(Hidden_Dimensions, Hidden_Dimensions)))
            else:
                larr_Layers.append(nn.Linear(Hidden_Dimensions, Hidden_Dimensions))
            larr_Layers.append(Activation())

        # Output layer
        if Layer_Initialization is not None:
            larr_Layers.append(Layer_Initialization(nn.Linear(Hidden_Dimensions, Output_Dimension)))
        else:
            larr_Layers.append(nn.Linear(Hidden_Dimensions, Output_Dimension))

        if (Final_Activation is not None):
            larr_Layers.append(Final_Activation())

        self.Layers = nn.Sequential(*larr_Layers)

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.Layers(X)