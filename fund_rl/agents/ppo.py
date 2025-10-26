from fund_rl.agents.base import TAgent
from fund_rl.buffers.trajectory import TTrajectory_Buffer
from fund_rl.models.actor_critic  import TActor_Critic_Network_Separate
import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

class TPPO_Agent(TAgent):
    """
    A Proximal Policy Optimization (PPO) agent implementation using PyTorch.
    Args:
        Environment: The environment in which the agent operates.
        Learning_Rate (float): The learning rate for the optimizer.
        Discount_Factor (float): The discount factor for future rewards.
        Hidden_Layers (int): Number of hidden layers in the neural network.
        Hidden_Dimensions (int): Number of neurons in each hidden layer.
        Batch_Size (int): The number of samples per training batch.
        Minibatch_Size (int): The number of samples per minibatch.
        Iterations (int): The number of training iterations per update.
        Policy_Clip (float): The clipping parameter for policy updates.
        Value_Factor (float): The factor for value loss in the total loss.
        Entropy_Factor (float): The factor for entropy regularization to encourage exploration.
        GAE_Lambda (float): The lambda parameter for Generalized Advantage Estimation.
    """
    def __init__(self, Environment, Learning_Rate=1e-3, Discount_Factor=0.99, Hidden_Layers=1, Hidden_Dimensions=128,
                 Batch_Size=512, Minibatch_Size=128, Iterations=4, Policy_Clip=0.2, Value_Factor=0.5,
                 Entropy_Factor=0.01, GAE_Lambda=0.95):
        super().__init__(Environment)
        self.Name = "PPO"

        # PPO specific parameters
        self.gf_Learning_Rate = Learning_Rate
        self.gf_Discount_Factor = Discount_Factor
        self.gi_Batch_Size = Batch_Size
        self.gi_Minibatch_Size = Minibatch_Size
        self.gi_Iterations = Iterations
        self.gf_Policy_Clip = Policy_Clip
        self.gf_Value_Factor = Value_Factor
        self.gf_Entropy_Factor = Entropy_Factor
        self.gf_GAE_Lambda = GAE_Lambda
        self.Loss = 0.0
        self.Entropy = 0.0
        
        # Memory to store trajectories
        self.Memory = TTrajectory_Buffer()
        
        # Actor-Critic Network
        self.Actor_Critic = TActor_Critic_Network_Separate(Output_Dimensions=self.Action_Space, 
                                                  Input_Dimensions=self.Observation_Space, 
                                                  Hidden_Dimensions=Hidden_Dimensions, 
                                                  Hidden_Layers=Hidden_Layers,
                                                  Activation=nn.Tanh)
        # Optimizer
        self.Optimizer = torch.optim.Adam(self.Actor_Critic.parameters(), lr=self.gf_Learning_Rate, eps=1e-5)

    def Calculate_GAE(self, Rewards, Values, Dones):
        """
        Calculate Generalized Advantage Estimation (GAE).
        Args:
            Rewards: Tensor of rewards.
            Values: Tensor of value estimates.
            Dones: Tensor indicating episode termination.
        Returns:
            A tuple of (advantages, returns).
        """
        # Get Batch Size
        li_N = len(Rewards)

        # Initialize advantages and last advantage
        ll_Advantages = torch.zeros(li_N)
        ll_Last_Advantage = 0

        # Calculate advantages using GAE
        for C1 in reversed(range(li_N)):
            # Compute non-terminal mask
            ll_Next_Non_Terminal = 1.0 - Dones[C1]

            # Compute delta
            ll_Delta = Rewards[C1] + self.gf_Discount_Factor * Values[C1 + 1] * ll_Next_Non_Terminal - Values[C1]

            # Compute advantage
            ll_Advantages[C1] = ll_Last_Advantage = ll_Delta + self.gf_Discount_Factor * self.gf_GAE_Lambda * ll_Next_Non_Terminal * ll_Last_Advantage

        # Compute returns
        ll_Returns = ll_Advantages + Values[:-1]
        ll_Returns = ll_Returns.detach()

        # Normalize advantages
        ll_Advantages = (ll_Advantages - ll_Advantages.mean()) / (ll_Advantages.std() + 1e-8)

        return ll_Advantages, ll_Returns

    def Choose_Action(self, State):
        """
        Choose an action based on the current state.
        Args:
            State: The current state.   
        Returns:
            The chosen action.
        """
        with torch.no_grad():
            # Convert state to tensor
            ll_State = torch.tensor(State, dtype=torch.float32)

            # Get action logits
            ll_Logits = self.Actor_Critic(ll_State)[0]

            # Create categorical distribution
            ll_Action_Probabilities = Categorical(logits=ll_Logits)

            # Sample action
            li_Action = ll_Action_Probabilities.sample().item()

        return li_Action
    
    def Policy(self, State) -> np.ndarray:
        """
        Given a state, return the action probabilities.
        Args:
            State: The current state.
        Returns:
            A numpy array of action probabilities.
        """
        # Convert input to tensor
        ll_State = torch.tensor(State, dtype=torch.float32)
        
        # Ensure batch dimension
        if ll_State.ndim == 1:
            ll_State = ll_State.unsqueeze(0)
        
        # Get action probabilities
        larr_Probabilities = torch.softmax(self.Actor_Critic(ll_State)[0], dim=-1).detach().numpy()
        
        # If single state, return 1D array
        if larr_Probabilities.shape[0] == 1:
            return larr_Probabilities.squeeze(0)
        
        return larr_Probabilities

    def Update(self, Transition):
        """
        Update the agent from experience.
        Args:
            Transition: A tuple containing (state, action, reward, next_state, done).
        """
        # Only update during training
        if not self.Is_Training:
            return
        
        # Store the transition in memory
        self.Memory.Remember(Transition)
        
        # If not enough samples, return
        if len(self.Memory) < self.gi_Batch_Size:
            return
        
        # Train the agent
        self.Train()

        # Clear memory after training
        self.Memory.Clear()

    def Train(self):
        """
        Train the agent using the collected experience.
        """
        # Prepare batches from memory
        ll_States = torch.tensor(np.array([s for (s, _, _, _, _) in self.Memory.Trajectory()]), dtype=torch.float32)
        ll_Actions = torch.tensor(np.array([[a] for (_, a, _, _, _) in self.Memory.Trajectory()]), dtype=torch.long)
        ll_Rewards = torch.tensor(np.array([r for (_, _, r, _, _) in self.Memory.Trajectory()]), dtype=torch.float32)
        ll_Dones = torch.tensor(np.array([d for (_, _, _, _, d) in self.Memory.Trajectory()]), dtype=torch.float32)
        ll_Next_States = torch.tensor(np.array([s for (_, _, _, s, _) in self.Memory.Trajectory()]), dtype=torch.float32)
        
        with torch.no_grad():
            # Get value estimates for states and next states
            ll_Values = self.Actor_Critic(ll_States)[1].squeeze()
            lf_Last_Next_Value = self.Actor_Critic(ll_Next_States[-1].unsqueeze(0))[1].squeeze()
            ll_Values = torch.cat([ll_Values, lf_Last_Next_Value.unsqueeze(0)], dim=0)

        # Compute advantages and returns
        ll_Advantages, ll_Returns = self.Calculate_GAE(ll_Rewards, ll_Values, ll_Dones)

        # Create indices for sampling
        larr_Indices = np.arange(len(self.Memory))

        # Get old action logits
        ll_Old_Action_Logits = self.Actor_Critic(ll_States)[0]

        # Create categorical distribution for old actions
        ll_Old_Action_Probabilities = Categorical(logits=ll_Old_Action_Logits)

        # Get log probabilities of old actions
        ll_Old_Log_Action_Probabilities = ll_Old_Action_Probabilities.log_prob(ll_Actions.squeeze(1)).detach()

        # Reset loss and entropy
        self.Loss = 0.0
        self.Entropy = 0.0

        # Training loop
        for C1 in range(self.gi_Iterations):
            # Shuffle indices
            np.random.shuffle(larr_Indices)

            # Minibatch training
            for C2 in range(0, len(self.Memory), self.gi_Minibatch_Size):

                # Get minibatch indices
                ll_Indices = larr_Indices[C2:C2 + self.gi_Minibatch_Size]

                # Get minibatch data
                ll_Batch_States = ll_States[ll_Indices]
                ll_Batch_Actions = ll_Actions[ll_Indices]
                ll_Batch_Returns = ll_Returns[ll_Indices]
                ll_Batch_Advantages = ll_Advantages[ll_Indices]

                # Get current action logits
                ll_Action_Logits = self.Actor_Critic(ll_Batch_States)[0]

                # Create categorical distribution for current actions
                ll_Action_Probabilities = Categorical(logits=ll_Action_Logits)

                # Get log probabilities of current actions
                ll_Log_Action_Probabilities = ll_Action_Probabilities.log_prob(ll_Batch_Actions.squeeze(1))

                # Calculate ratio
                ll_Ratio = torch.exp(ll_Log_Action_Probabilities - ll_Old_Log_Action_Probabilities[ll_Indices])

                # Normalize batch advantages
                ll_Batch_Advantages_Normalized = (ll_Batch_Advantages - ll_Batch_Advantages.mean()) / (ll_Batch_Advantages.std() + 1e-8)
                
                # Compute ratio loss
                ll_Loss_1 = -ll_Ratio * ll_Batch_Advantages_Normalized

                # Compute ratio clipped loss
                ll_Loss_2 = -torch.clamp(ll_Ratio, 1 - self.gf_Policy_Clip, 1 + self.gf_Policy_Clip) * ll_Batch_Advantages_Normalized

                # Get value estimates for batch states
                ll_Values = self.Actor_Critic(ll_Batch_States)[1].squeeze(-1)

                # Compute value loss
                ll_Value_Loss = F.mse_loss(ll_Values, ll_Batch_Returns)

                # Compute entropy
                ll_Entropy = ll_Action_Probabilities.entropy().mean()

                # Total loss
                ll_Loss = torch.mean(torch.max(ll_Loss_1, ll_Loss_2)) + self.gf_Value_Factor * ll_Value_Loss - self.gf_Entropy_Factor * ll_Entropy

                # Backpropagation and optimization step
                self.Optimizer.zero_grad()
                ll_Loss.backward()
                self.Optimizer.step()

                # Accumulate loss
                self.Loss += ll_Loss.item()

                # Accumulate entropy
                self.Entropy += ll_Entropy.item()

    def __str__(self):
        ls_Result = TAgent.__str__(self)
        ls_Result += f"Learning Rate: {self.gf_Learning_Rate}\n"
        ls_Result += f"Discount Factor: {self.gf_Discount_Factor}\n"
        ls_Result += f"Batch Size: {self.gi_Batch_Size}\n"
        ls_Result += f"Minibatch Size: {self.gi_Minibatch_Size}\n"
        ls_Result += f"Iterations: {self.gi_Iterations}\n"
        ls_Result += f"Policy Clip: {self.gf_Policy_Clip}\n"
        ls_Result += f"Value Loss Weight: {self.gf_Value_Factor}\n"
        ls_Result += f"Entropy Loss Weight: {self.gf_Entropy_Factor}\n"
        ls_Result += f"Actor-Critic Network: {self.Actor_Critic}\n"
        ls_Result += f"Loss: {self.Loss}\n"
        ls_Result += f"Entropy: {self.Entropy}\n"
        ls_Result += f"Optimizer: {self.Optimizer}\n"
        ls_Result += f"Memory: {self.Memory}\n"
        return ls_Result