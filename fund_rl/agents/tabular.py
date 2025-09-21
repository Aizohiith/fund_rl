from fund_rl.agents.base import TAgent
import numpy as np
import json

class TTabular_Agent(TAgent):
    def __init__(self, Environment, Learning_Rate=0.1, Discount_Factor=0.99, Initial_Q_Value=0.0):
        super().__init__(Environment)
        self.gf_Learning_Rate = Learning_Rate
        self.gf_Discount_Factor = Discount_Factor
        self.gf_Initial_Q_Value = Initial_Q_Value
        self.Q_Table = {}
        self.Name = "Tabular [Not Set]"
        
    
    def Save(self, ps_Path):
        readable=True
        if readable:
            # Convert NumPy arrays in Q_Table to lists for JSON compatibility
            json_compatible_q_table = {str(k): v.tolist() if isinstance(v, np.ndarray) else v
                                    for k, v in self.Q_Table.items()}
            with open(ps_Path + ".json", "w") as f:
                json.dump(json_compatible_q_table, f, indent=4)
        else:
            np.save(ps_Path + ".npy", self.Q_Table, allow_pickle=True)
    
    def Load(self, ps_Path):
        self.Q_Table = np.load(ps_Path + ".np")
    
    def __str__(self):
        ls_Result = TAgent.__str__(self)
        ls_Result += "Q Table:\n"
        ls_Result += f"\tLearning Rate: {self.gf_Learning_Rate}\n"
        ls_Result += f"\tDiscount Factor: {self.gf_Discount_Factor}\n"
        ls_Result += f"\tInitial Q-Value: " + str(self.gf_Initial_Q_Value) + "\n"
        ls_Result += f"\tStates: {len(self.Q_Table)}\n"
        return ls_Result