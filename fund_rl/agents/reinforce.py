from fund_rl.agents.base import TAgent
from fund_rl.buffers.trajectory import TTrajectory_Buffer
from fund_rl.models.network  import TBasic_Network
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class TReinforce_Agent(TAgent):
    """ 
    REINFORCE Agent implementation using PyTorch.
    Arguments:
        Environment: The environment in which the agent operates.
        Learning_Rate (float): The learning rate for the optimizer.
        Discount_Factor (float): The discount factor for future rewards.
        Entropy_Factor (float): The factor for entropy regularization to encourage exploration.
        Hidden_Layers (int): Number of hidden layers in the neural network.
        Hidden_Dimensions (int): Number of neurons in each hidden layer.
    """
    def __init__(self, Environment, 
                 Learning_Rate : float = 0.001, 
                 Discount_Factor : float = 0.99, 
                 Entropy_Factor : float = 0.001,
                 Hidden_Layers : int = 1, 
                 Hidden_Dimensions : int = 128):
        super().__init__(Environment)
        self.Name = "REINFORCE"
        self.Entropy = 0.0
        self.Loss = 0.0
        self.gf_Learning_Rate = Learning_Rate
        self.gf_Discount_Factor = Discount_Factor
        self.gf_Entropy_Factor = Entropy_Factor
        self.Memory = TTrajectory_Buffer()
        self.Model = TBasic_Network(Output_Dimension=self.Action_Space, 
                                    Input_Dimension=self.Observation_Space, 
                                    Hidden_Dimensions=Hidden_Dimensions, 
                                    Hidden_Layers=Hidden_Layers,
                                    Activation=nn.ReLU)
        
        self.Optimizer = torch.optim.RMSprop(self.Model.parameters(), lr=self.gf_Learning_Rate)

    def Calculate_Return(self):
        """
        Calculate the returns for each step in the trajectory.
        Returns:
            A numpy array of returns.
        """
        #create an array to hold the returns
        larr_Returns = np.zeros(len(self.Memory), dtype=np.float32)

        #for each step in the trajectory, calculate the return
        lf_Return = 0.0
        for C1 in reversed(range(len(self.Memory))):
            #Return = reward + discount_factor * next_return
            lf_Return = self.Memory.Trajectory(C1)[2] + self.gf_Discount_Factor * lf_Return
            larr_Returns[C1] = lf_Return

        #e.g rewards = [1, 1, 1], discount factor = 0.99
        #returns = [1 + 0.99 * (1 + 0.99 * 1), 1 + 0.99 * 1, 1] = [2.9701, 1.99, 1]

        #normalize the returns
        larr_Returns -= np.mean(larr_Returns)
        larr_Returns /= np.std(larr_Returns) + 1e-8

        return larr_Returns

    def Choose_Action(self, State : np.ndarray) -> int:
        """
        Choose an action based on the current state.
        Args:
            State: The current state.   
        Returns:
            The chosen action.
        """
        # Convert state to a tensor
        ll_State = torch.tensor(State, dtype=torch.float32)

        # Get the logits from the model
        larr_Logits = self.Model(ll_State).detach()

        # generate a categorical distribution over the list of logits
        larr_Action_Probabilities = Categorical(logits=larr_Logits)

        # Sample an action using the distribution
        li_Action = larr_Action_Probabilities.sample().item()

        return li_Action

    def Update(self, Transition):
        """
        Update the agent from experience.
        Args:
            Transition: The transition data to update the agent.
        """
        # Check if agent is in training mode
        if (not self.Is_Training):
            return
        
        # Unpack the transition
        _, _, _, _, lb_Done = Transition

        # Store the transition in memory
        self.Memory.Remember(Transition)

        # If the episode is not done, return
        if (not lb_Done):
            return
        
        # If the episode is done, train the agent
        self.Train()

        # Clear the memory after training
        self.Memory.Clear()

    def Train(self):
        """
        Train the agent using the REINFORCE algorithm.
        """
        # Prepare the data from the trajectory
        ll_States = torch.tensor(np.array([s for (s, _, _, _, _) in self.Memory.Trajectory()]), dtype=torch.float32)
        ll_Actions = torch.tensor(np.array([[a] for (_, a, _, _, _) in self.Memory.Trajectory()]), dtype=torch.long)
        ll_Returns = torch.tensor(self.Calculate_Return(), dtype=torch.float32)

        # Get the logits from the model
        ll_Logits = self.Model(ll_States)

        # Get probabilities of actions taken
        ll_Action_Probabilities = Categorical(logits=ll_Logits)

        # Calculate log probabilities
        ll_Log_Action_Probabilities = ll_Action_Probabilities.log_prob(ll_Actions.squeeze())

        # Calculate the entropy
        ll_Entropy = ll_Action_Probabilities.entropy().mean()
        self.Entropy = ll_Entropy.item()

        # Calculate a baseline
        ll_Baseline = torch.mean(ll_Returns)

        # Calculate advantage estimates using the baseline
        ll_Advantage = ll_Returns - ll_Baseline

        # Calculate the loss
        ll_Loss = -torch.mean(ll_Advantage *  ll_Log_Action_Probabilities) - self.gf_Entropy_Factor * ll_Entropy

        # Perform backpropagation and optimization
        self.Optimizer.zero_grad()
        ll_Loss.backward()
        self.Optimizer.step()

        # Log the loss
        self.Loss = ll_Loss.item()

    def Policy(self, State: np.ndarray) -> np.ndarray:
        """
        Get the action probabilities for the given state.
        Args:
            State: The current state.
        Returns:
            A list of action probabilities.
        """
        # Convert input to tensor
        ll_State = torch.tensor(State, dtype=torch.float32)
        
        # Ensure batch dimension
        if ll_State.ndim == 1:
            ll_State = ll_State.unsqueeze(0)
        
        # Get action probabilities
        larr_Probabilities = torch.softmax(self.Model(ll_State), dim=-1).detach().numpy()
        
        # If single state, return 1D array
        if larr_Probabilities.shape[0] == 1:
            return larr_Probabilities.squeeze(0)
        
        return larr_Probabilities

    def __str__(self):
        """
        String representation of the agent.
        Returns:
            str: String representation of the agent.
        """
        ls_Result = TAgent.__str__(self)
        ls_Result += f"\n - Learning Rate: {self.gf_Learning_Rate}"
        ls_Result += f"\n - Discount Factor: {self.gf_Discount_Factor}"
        ls_Result += f"\n - Entropy Factor: {self.gf_Entropy_Factor}"
        ls_Result += f"\n - Model: {self.Model}"
        ls_Result += f"\n - Optimizer: {self.Optimizer}"
        ls_Result += str(self.Memory)
        return ls_Result