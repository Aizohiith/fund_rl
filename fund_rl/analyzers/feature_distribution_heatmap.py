from fund_rl.analyzers.base import TAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import copy
from fund_rl.utility.format import Format

class TFeature_Distribution_Heatmap_Analyzer(TAnalyzer):
    """
    Feature Distribution Heatmap Analyzer
    An analyzer that visualizes the distribution of features in the state space
    using heatmaps. It employs PCA to reduce the dimensionality of the state space
    for visualization purposes.
    Parameters:
        Feature_Indices (list): List of feature indices to analyze.
        Feature_Names (list): List of feature names corresponding to the indices.
        Bins (int): Number of bins for the heatmap histograms.
        Tracker (TMetric_Tracker): Optional tracker instance to collect state data.
    """
    def __init__(self, Feature_Indices=None, Feature_Names=None, Bins=50, Tracker=None):
        super().__init__(Tracker=Tracker)

        self.garr_Feature_Indices = copy(Feature_Indices) if Feature_Indices is not None else []
        self.garr_Feature_Names = copy(Feature_Names) if Feature_Names is not None else []
        self.gi_Bins = Bins

    def Set_Tracker(self, Tracker):
        super().Set_Tracker(Tracker)

        self.gg_Tracker.Add_Property("State")

    def Analyze(self):
        """
        Analyze the state feature distributions and generate heatmaps.
        """
        if self.gg_Tracker is None:
            raise ValueError("Tracker not set. Please set a TMetric_Tracker instance using Set_Tracker method.")

        larr_States = np.array(self.gg_Tracker.Data("State"))
        ll_PCA = PCA(n_components=2)
        self.gg_Report["State_Visitation_Heatmap"] = ll_PCA.fit_transform(larr_States)
    
    def Plot(self):
        """
        Plot the feature distribution heatmaps.
        """
        if not self.gg_Report:
            raise ValueError("No analysis report found. Please run Analyze method first.")
        
        ll_X, ll_Y = self.gg_Report["State_Visitation_Heatmap"][:, 0] , self.gg_Report["State_Visitation_Heatmap"][:, 1]

        plt.hist2d(ll_X, ll_Y, bins=self.gi_Bins, cmap='viridis')
        plt.colorbar(label="State Density")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("State Visitation Heatmap (PCA Projection)")
        plt.show()

        if not(self.garr_Feature_Indices and self.garr_Feature_Names):
            return
        
        larr_States = np.array(self.gg_Tracker.Data("State"))
        
        li_Features = len(self.garr_Feature_Indices)
        # Create a grid of subplots (heatmaps for each pair of features)
        ll_Figure, ll_Axies = plt.subplots(li_Features, li_Features, figsize=(15, 12))

        for C1 in range(li_Features):
            for C2 in range(li_Features):
                li_Feature_1 = self.garr_Feature_Indices[C1]
                li_Feature_2 = self.garr_Feature_Indices[C2]

                ll_Axis = ll_Axies[C1, C2]

                if C1 != C2:
                    Feature_1 = [state[li_Feature_1] for state in larr_States]
                    Feature_2 = [state[li_Feature_2] for state in larr_States]
                    c = ll_Axis.hist2d(Feature_1, Feature_2, bins=self.gi_Bins, cmap='viridis')
                    ll_Axis.set_title(f"{self.garr_Feature_Names[C1]} vs {self.garr_Feature_Names[C2]}")
                    ll_Figure.colorbar(c[3], ax=ll_Axis)  # Add colorbar for the 2D histogram
                    ll_Axis.set_xlabel(self.garr_Feature_Names[C1])
                    ll_Axis.set_ylabel(self.garr_Feature_Names[C2])
                else:
                    ll_Axis.hist([state[li_Feature_1] for state in larr_States], bins=self.gi_Bins)
                    ll_Axis.set_title(f"Distribution of {self.garr_Feature_Names[C1]}")
                    ll_Axis.set_xlabel(self.garr_Feature_Names[C1])
                    ll_Axis.set_ylabel("Frequency")

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def Print(self):
        """
        Print the feature distribution heatmap analysis report.
        This analyzer does not generate a textual report.
        """
        larr_Report = ["Feature Distribution Heatmap Analyzer Report:",
                       "Feature Distribution Heatmap Analyzer", 
                       "does not have a textual report.",
                       "Please use the Plot method to visualize the results."]
        print(Format(larr_Report))