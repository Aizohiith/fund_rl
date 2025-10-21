from fund_rl.trackers.metric import TMetric_Tracker

class TAnalyzer:
    def __init__(self, Tracker=None):
        self.gg_Report = {}
        self.gg_Tracker = None

        if Tracker is not None:
            self.Set_Tracker(Tracker)

    def Analyze(self):
        raise NotImplementedError("Analyze method must be implemented by subclasses")
    
    def Report(self):
        return self.gg_Report
    
    def Plot(self):
        raise NotImplementedError("Plot method must be implemented by subclasses")
    
    def Print(self):
        raise NotImplementedError("Print method must be implemented by subclasses")
    
    def Set_Tracker(self, Tracker):
        if not isinstance(Tracker, TMetric_Tracker):
            raise ValueError("Tracker must be an instance of TMetric_Tracker")
        self.gg_Tracker = Tracker