from fund_rl.analyzers.base import TAnalyzer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from fund_rl.utility.format import Format

class TState_Visitation_Analyzer(TAnalyzer):
    def __init__(self, Precision=2, Most_Common_States=10, Tracker=None):
        super().__init__()

        self.gf_Precision = Precision  # Precision for state representation
        self.gi_Most_Common_States = Most_Common_States  # Number of most common states to display

        if Tracker is not None:
            self.Set_Tracker(Tracker)

    def Set_Tracker(self, Tracker):
        super().Set_Tracker(Tracker)

        self.gg_Tracker.Add_Property("State")

    def Analyze(self):
        if self.gg_Tracker is None:
            raise ValueError("Tracker not set. Please set a TMetric_Tracker instance using Set_Tracker method.")

        larr_States = np.round(self.gg_Tracker.Data("State"), self.gf_Precision)
        if np.ndim(larr_States) > 1:
            larr_States = [tuple(ll_State) for ll_State in larr_States]

        li_State_Visit_Counts = Counter(larr_States)
        
        larr_Top_States, larr_Top_States_Count = zip(*li_State_Visit_Counts.most_common(self.gi_Most_Common_States))

        self.gg_Report['Top_States'] = larr_Top_States
        self.gg_Report['Top_States_Count'] = larr_Top_States_Count
        self.gg_Report['Unique_States'] = int(len(li_State_Visit_Counts))
        self.gg_Report['Total_Visits'] = int(sum(larr_Top_States_Count))
        self.gg_Report['Average_Visits_Per_State'] = np.mean(larr_Top_States_Count)
        self.gg_Report['Median_Visits_Per_State'] = np.median(larr_Top_States_Count)
        self.gg_Report['STD_Visits_Per_State'] = np.std(larr_Top_States_Count)
        
        return self.gg_Report
    
    def Plot(self):
        if not self.gg_Report:
            raise ValueError("No analysis report found. Please run Analyze method first.")
        
        plt.bar(range(len(self.gg_Report['Top_States'])), self.gg_Report['Top_States_Count'])
        plt.title("Top State Visit Counts")
        plt.xlabel("States")
        plt.ylabel("Visit Counts")
        plt.xticks(range(len(self.gg_Report['Top_States'])), self.gg_Report['Top_States'], rotation=45)
        plt.tight_layout()
        plt.show()

    def Print(self):
        if not self.gg_Report:
            raise ValueError("No analysis report found. Please run Analyze method first.")
        
        larr_Report = ["State Visitation Analysis Report:"]
        for ls_Key, ll_Value in self.gg_Report.items():
            if isinstance(ll_Value, (list, tuple, np.ndarray)):
                larr_Report.append(ls_Key + ':')
                for E1 in ll_Value:
                    larr_Report.append(f" - {E1}")
            else:
                larr_Report.append(f"{ls_Key}: {ll_Value:.3f}")
        print(Format(larr_Report))