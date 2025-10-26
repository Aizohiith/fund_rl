from fund_rl.agents.base import TAgent
from fund_rl.action_selection.epsilon_greedy import TEpsilon_Greedy_Action_Selection
from fund_rl.buffers.replay import TReplay_Buffer
from fund_rl.models.network import TBasic_Network
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import numpy as np

class TDDQN_Agent(TAgent, TEpsilon_Greedy_Action_Selection):
    def __init__(self, Environment, Learning_Rate=0.001, Discount_Factor=0.99, Exploration_Rate=1.0, 
                 Exploration_Decay=0.995, Batch_Size=1024, Update_Frequency=32, Swap_Frequency=128, 
                 Hidden_Dimensions=128, Hidden_Layers=1, Activation=nn.ReLU):
        TAgent.__init__(self, Environment)
        TEpsilon_Greedy_Action_Selection.__init__(self, Exploration_Rate, Exploration_Decay)
        self.Name = "DDQN"

        # DDQN specific parameters
        self.gf_Learning_Rate = Learning_Rate
        self.gf_Swap_Frequency = Swap_Frequency
        self.gf_Update_Frequency = Update_Frequency
        self.gf_Discount_Factor = Discount_Factor
        self.Memory = TReplay_Buffer(Batch_Size=Batch_Size)

        # Neural network models
        self.Model_Current = TBasic_Network(self.Observation_Space, self.Action_Space, 
                                            Hidden_Dimensions=Hidden_Dimensions, Hidden_Layers=Hidden_Layers,
                                            Activation=Activation)
        self.Model_Target = TBasic_Network(self.Observation_Space, self.Action_Space, 
                                           Hidden_Dimensions=Hidden_Dimensions, Hidden_Layers=Hidden_Layers,
                                           Activation=Activation)
        self.Model_Target.load_state_dict(self.Model_Current.state_dict())

        # Optimizer
        self.Optimizer = optim.RMSprop(self.Model_Current.parameters(), lr=self.gf_Learning_Rate)

    def Choose_Action(self, State):

        # Convert state to tensor
        ll_State = torch.tensor(State, dtype=torch.float32)
        li_Action = -1

        if self.Should_Explore() and self.Is_Training:
            # Choose the random action
            li_Action = np.random.choice(self.Action_Space)
        else:
            # Choose the best action
            li_Action = np.argmax(self.Model_Current.forward(ll_State).detach().numpy())

        return li_Action

    def Update(self, Transition):
        # Unpack the transition tuple
        ll_State, li_Action, lf_Reward, ll_Next_State, lb_Done = Transition

        # Only update if training
        if not (self.Is_Training):
            return

        # Store the transition in memory
        self.Memory.Remember((ll_State, li_Action, lf_Reward, ll_Next_State, lb_Done))

        # Check if it's time to update
        if (self.Training_Steps % self.gf_Update_Frequency == 0):
            self.Train()

        # Check if it's time to swap target network
        if (self.Training_Steps % self.gf_Swap_Frequency == 0):
            self.Model_Target.load_state_dict(self.Model_Current.state_dict())

        # If episode is done, decay exploration rate
        if lb_Done:
            self.Decay_Exploration_Rate()

    def Train(self):

        # Sample a batch from memory
        larr_Batch = self.Memory.Batch()

        # Convert batch elements into tensors
        ll_States = torch.tensor(np.array([s for (s, _, _, _, _) in larr_Batch]), dtype=torch.float32)
        li_Actions = torch.tensor(np.array([[a] for (_, a, _, _, _) in larr_Batch]), dtype=torch.long)  # Shape (batch_size, 1)
        lf_Rewards = torch.tensor(np.array([r for (_, _, r, _, _) in larr_Batch]), dtype=torch.float32)
        ll_Next_States = torch.tensor(np.array([s_ for (_, _, _, s_, _) in larr_Batch]), dtype=torch.float32)
        lb_Dones = torch.tensor(np.array([d for (_, _, _, _, d) in larr_Batch]), dtype=torch.float32)

        # Get Q-values for current states
        ll_Q_Values = self.Model_Current(ll_States)

        # Get Q-values for selected actions
        ll_Current_Q = ll_Q_Values.gather(1, li_Actions).squeeze(1)

        with torch.no_grad():
            # Compute q-values for next states using current network
            ll_Next_Q_Values_Current = self.Model_Current(ll_Next_States)

            # Determine next actions using current network
            li_Next_Actions = torch.argmax(ll_Next_Q_Values_Current, dim=1, keepdim=True)

            # Use target network to get Q-values for next states
            ll_Next_Q_Values = self.Model_Target(ll_Next_States)

            # Compute target Q-values
            ll_Target_Q = lf_Rewards + (1 - lb_Dones) * self.gf_Discount_Factor * ll_Next_Q_Values.gather(1, li_Next_Actions).squeeze(1)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(ll_Current_Q, ll_Target_Q)

        # Gradient descent step
        self.Optimizer.zero_grad()
        loss.backward()
        self.Optimizer.step()

        # Store loss for monitoring
        self.Loss = loss.item()

    def Policy(self, State):
        # Convert state to tensor
        ll_State = torch.tensor(State, dtype=torch.float32)

        # Initialize the probabilities
        larr_Probabilities = np.zeros(self.Action_Space)

        # Get the Q values for the state
        larr_Q_Values = self.Model_Current.forward(ll_State).detach().numpy()
 
        # Get the maximum Q value and its count
        lf_Max_Q_Value = np.max(larr_Q_Values)
        li_Max_Count = len(np.where(larr_Q_Values == lf_Max_Q_Value)[0])

        # Calculate the action probabilities
        if (self.Is_Training):
            larr_Probabilities = np.ones(self.Action_Space) * (self.Exploration_Rate / self.Action_Space)
            larr_Probabilities[np.where(larr_Q_Values == lf_Max_Q_Value)] = (1.0 / li_Max_Count) - self.Exploration_Rate + (li_Max_Count * self.Exploration_Rate / self.Action_Space)
        else:
            larr_Probabilities = np.zeros(self.Action_Space)
            larr_Probabilities[np.where(larr_Q_Values == lf_Max_Q_Value)] = (1.0 / li_Max_Count)
        
        # Return the action probabilities
        return larr_Probabilities

    def __str__(self):
        ls_Result = TAgent.__str__(self)
        ls_Result += TEpsilon_Greedy_Action_Selection.__str__(self)
        ls_Result += "DDQN:\n"
        ls_Result += f"\tDiscount Factor: {self.gf_Discount_Factor}\n"
        ls_Result += f"\tUpdate Frequency: {self.gf_Update_Frequency}\n"
        ls_Result += f"\tSwap Frequency: {self.gf_Swap_Frequency}\n"
        ls_Result += f"\tBatch Size: {self.Memory.gi_Batch_Size}\n"
        ls_Result += f"\tMemory Usage: {np.round(100 * len(self.Memory.Memory) / self.Memory.gi_Memory_Capacity )}\n"
        ls_Result += f"\tNetwork:\n"
        ls_Result += f"\t\t{self.Model_Current}\n"
        return ls_Result