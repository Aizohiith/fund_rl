from fund_rl.analyzers.base import TAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from fund_rl.utility.format import Format
from copy import copy

class TDecision_Boundary_Analyzer(TAnalyzer):
    def __init__(self, Environment, Agent, Samples = 10, Feature_Indices=None, Feature_Names=None, Feature_Minimum=None, Feature_Maximum=None, Tracker=None):
        super().__init__()

        self.garr_Feature_Indices = copy(Feature_Indices) if Feature_Indices is not None else []
        self.garr_Feature_Names = copy(Feature_Names) if Feature_Names is not None else []
        self.garr_Feature_Minimum = copy(Feature_Minimum) if Feature_Minimum is not None else []
        self.garr_Feature_Maximum = copy(Feature_Maximum) if Feature_Maximum is not None else []
        self.gg_Agent = Agent
        self.gg_Environment = Environment
        self.gi_Samples = Samples

        if Tracker is not None:
            self.Set_Tracker(Tracker)

    def Set_Tracker(self, Tracker):
        super().Set_Tracker(Tracker)

    def Analyze(self):
        if self.gg_Tracker is None:
            raise ValueError("Tracker not set. Please set a TMetric_Tracker instance using Set_Tracker method.")

        for C1 in range(len(self.garr_Feature_Indices) - 1):
            for C2 in range(C1 + 1, len(self.garr_Feature_Indices)):
                if C1 != C2:
                    li_Feature_1 = self.garr_Feature_Indices[C1]
                    li_Feature_2 = self.garr_Feature_Indices[C2]

                    # Create a grid of points covering the feature space
                    larr_x = np.linspace(self.garr_Feature_Minimum[C1], self.garr_Feature_Maximum[C1], 100)
                    larr_y = np.linspace(self.garr_Feature_Minimum[C2], self.garr_Feature_Maximum[C2], 100)
                    ll_x, ll_y = np.meshgrid(larr_x, larr_y)
                    ll_Grid_Points = np.c_[ll_x.ravel(), ll_y.ravel()]

                    # Predict actions for each point in the grid
                    ll_Actions = []
                    for point in ll_Grid_Points:
                        # Create a full state with zeros and set the two features
                        larr_State = np.zeros(self.gg_Environment.observation_space.shape[0])
                        larr_State[li_Feature_1] = point[0]
                        larr_State[li_Feature_2] = point[1]
                        li_Action = 0
                        for C3 in range(self.gi_Samples):
                            larr_State_Temp = copy(larr_State)
                            larr_State_Temp += np.random.normal(0, 0.1, size=larr_State.shape)  # Add some noise for robustness
                            li_Action += self.gg_Agent.Choose_Action(larr_State_Temp)

                        lf_Action = li_Action / self.gi_Samples
                        ll_Actions.append(lf_Action)
                    ll_Actions = np.array(ll_Actions).reshape(ll_x.shape)

                    # Store the decision boundary data
                    self.gg_Report[f"Decision_Boundary_{self.garr_Feature_Names[C1]}_vs_{self.garr_Feature_Names[C2]}"] = (ll_x, ll_y, ll_Actions)

        return self.gg_Report
    
    def Plot(self):
        if not self.gg_Report:
            raise ValueError("No analysis report found. Please run Analyze method first.")
        
        for key, (ll_X, ll_Y, ll_Actions) in self.gg_Report.items():
            plt.contourf(ll_X, ll_Y, ll_Actions, alpha=0.8, cmap='jet')
            plt.colorbar(label='Action')
            # Extract feature names from the key for correct labeling
            feature_names = key.replace("Decision_Boundary_", "").split("_vs_")
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
            plt.title(f"Decision Boundary: {key}")
            plt.show()

    def Print(self):
        if not self.gg_Report:
            raise ValueError("No analysis report found. Please run Analyze method first.")
        
        pass