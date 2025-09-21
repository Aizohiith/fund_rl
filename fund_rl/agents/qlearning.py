from fund_rl.agents.tabular import TTabular_Agent
from fund_rl.action_selection.epsilon_greedy import TEpsilon_Greedy_Action_Selection
from fund_rl.action_selection.ucb import TUCB_Action_Selection
from fund_rl.utility.index_state import Index_State
import numpy as np

class TQ(TTabular_Agent):
    def __init__(self, Environment, Learning_Rate=0.1, Discount_Factor=0.99, Initial_Q_Value=0.0):
        TTabular_Agent.__init__(self, Environment, Learning_Rate, Discount_Factor, Initial_Q_Value=Initial_Q_Value)
        self.gf_Exploration_Rate = 0.1
        self.Name = "Q-Learning"

    def Update(self, Transition):
        # Unpack the transition tuple
        State, Action, Reward, Next_State, Done = Transition
        # Index the current state
        ll_State = Index_State(State, self.Q_Table, self.Action_Space, self.gf_Initial_Q_Value)
        # Index the next state
        ll_Next_State = Index_State(Next_State, self.Q_Table, self.Action_Space, self.gf_Initial_Q_Value)
        # Choose the next action
        li_Action = Action
        li_Next_Action = np.argmax(self.Q_Table[ll_Next_State])
        # Calculate the future Q value
        lf_Future_Q_Value =  self.Q_Table[ll_Next_State][li_Next_Action] * (not Done)
        # Calculate the temporal difference
        lf_Temporal_Difference = Reward + self.gf_Discount_Factor * lf_Future_Q_Value - self.Q_Table[ll_State][li_Action] 
        # Update the Q table
        self.Q_Table[ll_State][li_Action] = self.Q_Table[ll_State][li_Action] + self.gf_Learning_Rate * lf_Temporal_Difference
        #Calculate the loss
        self.Loss = lf_Temporal_Difference ** 2

        # Increment the training steps
        self.Training_Steps += 1
        
    def __str__(self):
        ls_Result = TTabular_Agent.__str__(self)
        return ls_Result
    
class TQ_Epsilon_Agent(TQ, TEpsilon_Greedy_Action_Selection):
    def __init__(self, Environment, Learning_Rate=0.1, Discount_Factor=0.99, Initial_Q_Value=0.0, Exploration_Rate=1.0, Exploration_Decay=0.995, Min_Exploration_Rate=0.01):
        TQ.__init__(self, Environment, Learning_Rate, Discount_Factor, Initial_Q_Value=Initial_Q_Value)
        TEpsilon_Greedy_Action_Selection.__init__(self, Exploration_Rate, Exploration_Decay, Min_Exploration_Rate)
        self.Name = "Q-Learning [Epsilon Greedy]"

    def Choose_Action(self, State):
        # Index the current state
        ll_State = Index_State(State, self.Q_Table, self.Action_Space, self.gf_Initial_Q_Value)

        # Choose the random action
        if self.Should_Explore() and self.Is_Training:
            return np.random.choice(self.Action_Space)  # Choose a random action
        
        # Choose the best action
        return np.argmax(self.Q_Table[ll_State])
    
    def Update(self, Transition):
        super().Update(Transition)
        
        ll_State, li_Action, lf_Reward, ll_Next_State, lb_Done = Transition
        if lb_Done:
            # Decay the exploration rate
            self.Decay_Exploration_Rate()

    def __str__(self):
        ls_Result = TQ.__str__(self)
        ls_Result += TEpsilon_Greedy_Action_Selection.__str__(self)
        return ls_Result

    def Policy(self, State):
        ll_Indexed_State = Index_State(State, self.Q_Table, self.Action_Space, self.gf_Initial_Q_Value)
        larr_Q_Values = self.Q_Table[ll_Indexed_State]
        lf_Max_Q_Value = np.max(larr_Q_Values)
        if (self.Is_Training):
            larr_Probabilities = np.ones(self.Action_Space) * (self.Exploration_Rate / self.Action_Space)
            larr_Probabilities[np.where(larr_Q_Values == lf_Max_Q_Value)] = 1 - self.Exploration_Rate + (self.Exploration_Rate / self.Action_Space)
        else:
            larr_Probabilities = np.zeros(self.Action_Space)
            larr_Probabilities[np.where(larr_Q_Values == lf_Max_Q_Value)] = 1 / len(np.where(larr_Q_Values == lf_Max_Q_Value)[0])
        return larr_Probabilities
    
class TQ_UCB_Agent(TQ, TUCB_Action_Selection):
    def __init__(self, Environment, Learning_Rate=0.1, Discount_Factor=0.99, Initial_Q_Value=0.0, C=2.0, Precision=1):
        TQ.__init__(self, Environment, Learning_Rate, Discount_Factor, Initial_Q_Value=Initial_Q_Value)
        TUCB_Action_Selection.__init__(self, self.Action_Space, C, Precision=Precision)
        self.Name = "Q-Learning [UCB]"

    def Choose_Action(self, State):
        # Index the current state
        ll_State = Index_State(State, self.Q_Table, self.Action_Space, self.gf_Initial_Q_Value)

        if self.Is_Training:
            # Choose the action
            return np.argmax(self.Q_Table[ll_State] + self.Exploration_Values(State, self.Training_Steps))
        else:
            # Choose the best action
            return np.argmax(self.Q_Table[ll_State])

    def Update(self, Transition):
        super().Update(Transition)
        # Unpack the transition tuple
        ll_State, li_Action, lf_Reward, ll_Next_State, lb_Done = Transition

        self.Take_Action(ll_State, li_Action)


    def __str__(self):
        ls_Result = TQ.__str__(self)
        ls_Result += TUCB_Action_Selection.__str__(self)
        return ls_Result