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
        self.gf_Learning_Rate = Learning_Rate

        self.gf_Swap_Frequency = Swap_Frequency
        self.gf_Update_Frequency = Update_Frequency
        self.gf_Discount_Factor = Discount_Factor
        self.Memory = TReplay_Buffer(Batch_Size=Batch_Size)
        self.Model_Current = TBasic_Network(self.Observation_Space, self.Action_Space, 
                                            Hidden_Dimensions=Hidden_Dimensions, Hidden_Layers=Hidden_Layers,
                                            Activation=Activation)
        self.Model_Target = TBasic_Network(self.Observation_Space, self.Action_Space, 
                                           Hidden_Dimensions=Hidden_Dimensions, Hidden_Layers=Hidden_Layers,
                                           Activation=Activation)
        self.Model_Target.load_state_dict(self.Model_Current.state_dict())

        self.Optimizer = optim.RMSprop(self.Model_Current.parameters(), lr=self.gf_Learning_Rate)
        self.Name = "DDQN"

    def Choose_Action(self, State):

        ll_State = torch.tensor(State, dtype=torch.float32)
        li_Action = -1
        # Choose the random action
        if self.Should_Explore() and self.Is_Training:
            li_Action = np.random.choice(self.Action_Space)
        else:
            # Choose the best action
            li_Action = np.argmax(self.Model_Current.forward(ll_State).detach().numpy())

        return li_Action

    def Update(self, Transition):
        # Unpack the transition tuple
        ll_State, li_Action, lf_Reward, ll_Next_State, lb_Done = Transition
        if not (self.Is_Training):
            return

        self.Memory.Remember((ll_State, li_Action, lf_Reward, ll_Next_State, lb_Done))

        if (self.Training_Steps % self.gf_Swap_Frequency == 0):
            self.Model_Target.load_state_dict(self.Model_Current.state_dict())

        if (self.Training_Steps % self.gf_Update_Frequency == 0):
            self.Train()

        if lb_Done:
            self.Decay_Exploration_Rate()

    def Train(self):
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
        lf_Current_Q = ll_Q_Values.gather(1, li_Actions).squeeze(1)  # Shape (batch_size,)

        # Compute target Q-values
        with torch.no_grad():
            ll_Next_Q_Values = self.Model_Target(ll_Next_States)  # Compute Q-values for next states
            li_Next_Actions = torch.argmax(ll_Next_Q_Values, dim=1, keepdim=True)  # Best action for next state
            lf_Target_Q = lf_Rewards + (1 - lb_Dones) * self.gf_Discount_Factor * ll_Next_Q_Values.gather(1, li_Next_Actions).squeeze(1)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(lf_Current_Q, lf_Target_Q)

        # Gradient descent step
        self.Optimizer.zero_grad()
        loss.backward()
        self.Optimizer.step()

        # Store loss for monitoring
        self.Loss = loss.item()

    def Policy(self, State):
        raise NotImplementedError("Policy method not implemented for DDQN agent.")

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