from fund_rl.models.base import TNetwork
import torch
import torch.nn as nn
import copy
from typing import Optional, Callable, Type

class TActor_Critic_Network(TNetwork):
    """
    A shared-body Actor-Critic neural network used in reinforcement learning.

    This model uses a shared multi-layer perceptron (MLP) backbone, with two heads:
    - An actor head that outputs a policy over actions.
    - A critic head that outputs a scalar value estimate of the input state.

    Args:
        Input_Dimensions (int): Dimension of the input vector.
        Output_Dimensions (int): Dimension of the action space (policy output).
        Hidden_Dimensions (int, optional): Number of units in each hidden layer. Default is 128.
        Hidden_Layers (int, optional): Number of hidden layers. Default is 1.
        Final_Activation (Optional[Callable[[], nn.Module]]): Final activation function for actor (e.g., nn.Softmax).
        Activation (Type[nn.Module], optional): Activation function used in hidden layers. Default is nn.Tanh.
        Layer_Initialization (Optional[Callable[[nn.Module], nn.Module]]): Optional layer initializer (e.g., Xavier).
    """
    def __init__(self, Input_Dimensions : int, Output_Dimensions : int, Hidden_Dimensions : int = 128, 
                 Hidden_Layers : int = 1, Final_Activation : Optional[Callable[[], nn.Module]] = None, Activation : Type[nn.Module] = nn.Tanh,
                 Layer_Initialization : Optional[Callable[[nn.Module], nn.Module]] = None) -> None:
        super().__init__(Input_Dimensions=Input_Dimensions, Output_Dimensions=Output_Dimensions)

        larr_Layers = []

        # Input layer
        if Layer_Initialization is not None:
            larr_Layers.append(Layer_Initialization(nn.Linear(Input_Dimensions, Hidden_Dimensions)))
        else:
            larr_Layers.append(nn.Linear(Input_Dimensions, Hidden_Dimensions))
        larr_Layers.append(Activation())

        # Hidden layers
        for C1 in range(Hidden_Layers):
            if Layer_Initialization is not None:
                larr_Layers.append(Layer_Initialization(nn.Linear(Hidden_Dimensions, Hidden_Dimensions)))
            else:
                larr_Layers.append(nn.Linear(Hidden_Dimensions, Hidden_Dimensions))
            larr_Layers.append(Activation())

        # Shared layers
        self.Shared = nn.Sequential(*larr_Layers)

        # Critic layers
        if Layer_Initialization is not None:
            self.Critic = Layer_Initialization(nn.Linear(Hidden_Dimensions, 1))
        else:
            self.Critic = nn.Linear(Hidden_Dimensions, 1)

        # Actor layers
        if Layer_Initialization is not None:
            self.Actor = Layer_Initialization(nn.Linear(Hidden_Dimensions, Output_Dimensions))
        else:
            self.Actor = nn.Linear(Hidden_Dimensions, Output_Dimensions)

        if (Final_Activation is None):
            return

        if Final_Activation == nn.Softmax:
            self.Actor = nn.Sequential(self.Actor, Final_Activation(dim=-1))
        else:
            self.Actor = nn.Sequential(self.Actor, Final_Activation())

    def forward(self, X : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - ll_Actor (torch.Tensor): Output from the actor head (policy).
                - ll_Critic (torch.Tensor): Output from the critic head (value estimate).
        """
        # Shared Output
        ll_Actor_Critic = self.Shared(X)

        # Critic Output
        ll_Critic = self.Critic(ll_Actor_Critic)

        # Actor Output
        ll_Actor = self.Actor(ll_Actor_Critic)
        return ll_Actor, ll_Critic
    
class TActor_Critic_Network_Separate(TNetwork):
    """
    A dual-body Actor-Critic neural network for reinforcement learning.

    This network architecture uses separate MLP backbones for the actor and critic,
    allowing for independent learning of the policy and value functions. This is
    useful when the actor and critic may benefit from differing representations.

    Args:
        Input_Dimensions (int): Dimension of the input feature vector.
        Output_Dimensions (int): Dimension of the output action space (policy output).
        Hidden_Dimensions (int, optional): Number of units in each hidden layer. Default is 128.
        Hidden_Layers (int, optional): Number of hidden layers. Default is 1.
        Final_Activation (Optional[Callable[[], nn.Module]], optional): Final activation for actor
            output layer (e.g., nn.Softmax for discrete actions). If None, no activation is applied.
        Activation (Type[nn.Module], optional): Activation function class used in hidden layers. Default is nn.Tanh.
        Layer_Initialization (Optional[Callable[[nn.Module], nn.Module]], optional): Optional initializer function
            to apply to linear layers (e.g., for custom weight initialization).
    """
    def __init__(self, Input_Dimensions : int, Output_Dimensions : int, Hidden_Dimensions : int = 128, 
                 Hidden_Layers : int = 1, Final_Activation : Optional[Callable[[], nn.Module]] = None, Activation : Type[nn.Module] = nn.Tanh,
                 Layer_Initialization : Optional[Callable[[nn.Module], nn.Module]] = None) -> None:        
        super().__init__(Input_Dimensions=Input_Dimensions, Output_Dimensions=Output_Dimensions)

        larr_Layers = []

        # Input layer
        if Layer_Initialization is not None:
            larr_Layers.append(Layer_Initialization(nn.Linear(Input_Dimensions, Hidden_Dimensions)))
        else:
            larr_Layers.append(nn.Linear(Input_Dimensions, Hidden_Dimensions))
        larr_Layers.append(Activation())

        # Hidden layers
        for C1 in range(Hidden_Layers):
            if Layer_Initialization is not None:
                larr_Layers.append(Layer_Initialization(nn.Linear(Hidden_Dimensions, Hidden_Dimensions)))
            else:
                larr_Layers.append(nn.Linear(Hidden_Dimensions, Hidden_Dimensions))
            larr_Layers.append(Activation())

        # Critic layers
        if Layer_Initialization is not None:
            self.Critic = nn.Sequential(*copy.deepcopy(larr_Layers), Layer_Initialization(nn.Linear(Hidden_Dimensions, 1)))
        else:
            self.Critic = nn.Sequential(*copy.deepcopy(larr_Layers), nn.Linear(Hidden_Dimensions, 1))

        # Actor layers
        if Layer_Initialization is not None:
            self.Actor = nn.Sequential(*copy.deepcopy(larr_Layers), Layer_Initialization(nn.Linear(Hidden_Dimensions, Output_Dimensions)))
        else:
            self.Actor = nn.Sequential(*copy.deepcopy(larr_Layers), nn.Linear(Hidden_Dimensions, Output_Dimensions))

        if (Final_Activation is None):
            return
        if Final_Activation == nn.Softmax:
            self.Actor = nn.Sequential(self.Actor, Final_Activation(dim=-1))
        else:
            self.Actor = nn.Sequential(self.Actor, Final_Activation())

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - ll_Actor (torch.Tensor): Output from the actor head (policy).
                - ll_Critic (torch.Tensor): Output from the critic head (value estimate).
        """
        # Critic Output
        ll_Critic = self.Critic(X)

        # Actor Output
        ll_Actor = self.Actor(X)
        return ll_Actor, ll_Critic