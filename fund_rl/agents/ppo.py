from fund_rl.agents.base import TAgent
from fund_rl.buffers.trajectory import TTrajectory_Buffer
from fund_rl.models.actor_critic  import TActor_Critic_Network_Separate
import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

class TPPO_Agent(TAgent):
    def __init__(self, Environment, Learning_Rate=1e-3, Discount_Factor=0.99, Hidden_Layers=1, Hidden_Dimensions=128,
                 Batch_Size=512, Minibatch_Size=128, Iterations=4, Value_Clip=0.2, Value_Factor=0.5,
                 Entropy_Factor=0.01):
        super().__init__(Environment)
        self.Name = "PPO"
        self.gf_Learning_Rate = Learning_Rate
        self.gf_Discount_Factor = Discount_Factor
        self.gi_Batch_Size = Batch_Size
        self.gi_Minibatch_Size = Minibatch_Size
        self.gi_Iterations = Iterations
        self.gf_Value_Clip = Value_Clip
        self.gf_Value_Factor = Value_Factor
        self.gf_Entropy_Factor = Entropy_Factor
        self.Loss = 0.0
        self.Entropy = 0.0
        
        self.Memory = TTrajectory_Buffer()
        
        self.Actor_Critic = TActor_Critic_Network_Separate(Output_Dimensions=self.Action_Space, 
                                                  Input_Dimensions=self.Observation_Space, 
                                                  Hidden_Dimensions=Hidden_Dimensions, 
                                                  Hidden_Layers=Hidden_Layers,
                                                  Activation=nn.Tanh)
        
        self.Optimizer = torch.optim.Adam(self.Actor_Critic.parameters(), lr=self.gf_Learning_Rate, eps=1e-5)

    def calculate_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (Tensor): Tensor of rewards (T,).
            values (Tensor): Tensor of value estimates (T+1,).
            dones (Tensor): Tensor indicating episode termination (T,), 1 if done, else 0.
            gamma (float): Discount factor.
            lambda_ (float): GAE lambda.

        Returns:
            advantages (Tensor): Computed GAE advantages (T,).
            returns (Tensor): Computed discounted returns (T,).
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        last_adv = 0

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_adv = delta + gamma * lambda_ * next_non_terminal * last_adv

        returns = advantages + values[:-1]  # value[:-1] since value has T+1 entries
        return advantages, returns
    def Calculate_Return(self, pp_Rewards, pp_Dones):

        larr_Returns = np.zeros(len(pp_Rewards), dtype=np.float32)
        lf_Return = 0.0
        for C1 in reversed(range(len(pp_Rewards))):
            lf_Return = pp_Rewards[C1] + self.gf_Discount_Factor * lf_Return * (1 - pp_Dones[C1])
            larr_Returns[C1] = lf_Return

        return larr_Returns
    
    def Calculate_Advantage(self, pp_States, pp_Returns):
        ll_Values = self.Actor_Critic(pp_States)[1].detach().squeeze(1)
        ll_Advantage = pp_Returns - ll_Values
        return ll_Advantage

    def Choose_Action(self, pp_State):
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(pp_State, np.ndarray):
                state_tensor = torch.tensor(pp_State, dtype=torch.float32)
            else:
                state_tensor = pp_State.float()

            # Add batch dimension: [1, 4, 84, 84]
            if state_tensor.ndim == 3:
                state_tensor = state_tensor.unsqueeze(0)

            ll_Logits = self.Actor_Critic(state_tensor)[0]
            ll_Action_Probabilities = Categorical(logits=ll_Logits)
            li_Action = ll_Action_Probabilities.sample().item()

        return li_Action
    
    def Policy(self, pp_State) -> np.ndarray:
        # Convert input to tensor
        ll_State = torch.tensor(pp_State, dtype=torch.float32)
        
        # Ensure batch dimension
        if ll_State.ndim == 1:
            ll_State = ll_State.unsqueeze(0)
        
        # Get action probabilities
        larr_Probabilities = torch.softmax(self.Actor_Critic(ll_State)[0], dim=-1).detach().numpy()
        
        # If single state, return 1D array
        if larr_Probabilities.shape[0] == 1:
            return larr_Probabilities.squeeze(0)
        
        return larr_Probabilities


    
    def Update(self, pp_Transition):
        if not self.Is_Training:
            return
        
        self.Memory.Remember(pp_Transition)
        
        if len(self.Memory) < self.gi_Batch_Size:
            return
        
        self.Train()
        self.Memory.Clear()

    def Train(self):
        
        ll_States = torch.tensor(np.array([s for (s, _, _, _, _) in self.Memory.Trajectory()]), dtype=torch.float32)
        ll_Actions = torch.tensor(np.array([[a] for (_, a, _, _, _) in self.Memory.Trajectory()]), dtype=torch.long)
        ll_Rewards = torch.tensor(np.array([r for (_, _, r, _, _) in self.Memory.Trajectory()]), dtype=torch.float32)
        ll_Dones = torch.tensor(np.array([d for (_, _, _, _, d) in self.Memory.Trajectory()]), dtype=torch.float32)

        ll_Next_States = torch.tensor(np.array([s for (_, _, _, s, _,) in self.Memory.Trajectory()]), dtype=torch.float32)
        
        #ll_Returns = torch.tensor(self.Calculate_Return(ll_Rewards, ll_Dones), dtype=torch.float32)
        #ll_Advantages = self.Calculate_Advantage(ll_States, ll_Returns)
        with torch.no_grad():
            values = self.Actor_Critic(ll_States)[1].squeeze()  # shape: [T]
            next_value = self.Actor_Critic(ll_Next_States[-1].unsqueeze(0))[1].squeeze()  # scalar value for last next state

            values = torch.cat([values, next_value.unsqueeze(0)], dim=0)  # shape: [T+1]

        ll_Advantages, ll_Returns = self.calculate_gae(ll_Rewards, values , ll_Dones)
        ll_Advantages = (ll_Advantages - ll_Advantages.mean()) / (ll_Advantages.std() + 1e-8)

        larr_Indices = np.arange(len(self.Memory))

        ll_Old_Action_Logits = self.Actor_Critic(ll_States)[0]
        ll_Old_Action_Probabilities = Categorical(logits=ll_Old_Action_Logits)
        ll_Old_Log_Action_Probabilities = ll_Old_Action_Probabilities.log_prob(ll_Actions.squeeze(1)).detach()

        self.Loss = 0.0
        self.Entropy = 0.0
        for C1 in range(self.gi_Iterations):
            np.random.shuffle(larr_Indices)

            for C2 in range(0, len(self.Memory), self.gi_Minibatch_Size):
                ll_Indices = larr_Indices[C2:C2 + self.gi_Minibatch_Size]
                ll_Batch_States = ll_States[ll_Indices]
                ll_Batch_Actions = ll_Actions[ll_Indices]
                ll_Batch_Returns = ll_Returns[ll_Indices]
                ll_Batch_Advantages = ll_Advantages[ll_Indices]

                ll_Action_Logits = self.Actor_Critic(ll_Batch_States)[0]
                ll_Action_Probabilities = Categorical(logits=ll_Action_Logits)
                ll_Log_Action_Probabilities = ll_Action_Probabilities.log_prob(ll_Batch_Actions.squeeze(1))

                ll_Ratio = torch.exp(ll_Log_Action_Probabilities - ll_Old_Log_Action_Probabilities[ll_Indices])

                ll_Batch_Advantages_Normalized = (ll_Batch_Advantages - ll_Batch_Advantages.mean()) / (ll_Batch_Advantages.std() + 1e-8)
                
                ll_Loss_1 = -ll_Ratio * ll_Batch_Advantages_Normalized
                ll_Loss_2 = -torch.clamp(ll_Ratio, 1 - self.gf_Value_Clip, 1 + self.gf_Value_Clip) * ll_Batch_Advantages_Normalized


                ll_Values = self.Actor_Critic(ll_Batch_States)[1].squeeze(-1)
                ll_Value_Loss = F.mse_loss(ll_Values, ll_Batch_Returns)

                ll_Entropy = ll_Action_Probabilities.entropy().mean()

                ll_Loss = torch.mean(torch.max(ll_Loss_1, ll_Loss_2)) + self.gf_Value_Factor * ll_Value_Loss - self.gf_Entropy_Factor * ll_Entropy

                self.Optimizer.zero_grad()
                ll_Loss.backward()
                self.Optimizer.step()

                self.Loss += ll_Loss.item()
                self.Entropy += ll_Entropy.item()


    def Save(self, ps_Path):
        """
        Save the agent's model to the specified path.
        Args:
            ps_Path: Path to save the model.
        """
        raise NotImplementedError("Save method not implemented.")
    
    def Load(self, ps_Path):
        """
        Load the agent's model from the specified path.
        Args:
            ps_Path: Path to load the model from.
        """
        raise NotImplementedError("Load method not implemented.")
    

    def __str__(self):
        ls_Result = TAgent.__str__(self)
        ls_Result += f"Learning Rate: {self.Learning_Rate}\n"
        ls_Result += f"Discount Factor: {self.Discount_Factor}\n"
        ls_Result += f"Batch Size: {self.Batch_Size}\n"
        ls_Result += f"Minibatch Size: {self.Minibatch_Size}\n"
        ls_Result += f"Iterations: {self.Iterations}\n"
        ls_Result += f"Value Clip: {self.Value_Clip}\n"
        ls_Result += f"Value Loss Weight: {self.Value_Loss_Weight}\n"
        ls_Result += f"Entropy Loss Weight: {self.Entropy_Loss_Weight}\n"
        ls_Result += f"Actor-Critic Network: {self.Actor_Critic}\n"
        ls_Result += f"Optimizer: {self.Optimizer}\n"
        ls_Result += f"Memory: {self.Memory}\n"
        return ls_Result