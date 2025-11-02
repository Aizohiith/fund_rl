from fund_rl.analyzers.base import TAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from fund_rl.utility.format import Format

class TPolicy_Visualisation_Analyzer(TAnalyzer):
    """
    Policy Visualization Analyzer for Reinforcement Learning Agents.
    This analyzer visualizes the policy of a given RL agent in a grid-based environment
    by plotting arrows that indicate the action probabilities for each state.
    Parameters:
        Environment (gym.Env): The environment in which the agent operates.
        Agent (TAgent): The RL agent whose policy is to be visualized.
        One_Hot (bool): Whether the environment uses one-hot encoded states.
        Action_Order (list): The order of actions corresponding to directions (e.g., [LEFT, DOWN, RIGHT, UP]).
        Grid_Size (tuple): The size of the grid environment (rows, columns).
        Arrow_Offset (float): The offset for arrow placement within each grid cell.
        Tracker (TMetric_Tracker, optional): An optional metric tracker for logging metrics.
    """
    def __init__(self, Environment, Agent, One_Hot=False, Action_Order = [1, 2, 3, 4], Grid_Size=(4, 4), Arrow_Offset=0.2, Tracker=None):
        super().__init__(Tracker=Tracker)

        self.gg_Agent = Agent
        self.gg_Environment = Environment
        self.gg_Grid_Size = Grid_Size
        self.gf_Arrow_Offset = Arrow_Offset
        self.garr_Action_Order = Action_Order
        self.gb_One_Hot = One_Hot

    def Analyze(self):
        """
        Analyze the agent's policy and prepare data for visualization.
        """
        self.gg_Report['Policy'] = []
        for li_State in range(self.gg_Environment.observation_space.n):
            if self.gb_One_Hot:
                larr_One_Hot_State = np.zeros(self.gg_Environment.observation_space.n)
                larr_One_Hot_State[li_State] = 1
                ll_State = larr_One_Hot_State
            else:
                ll_State = li_State
            li_Row = self.gg_Grid_Size[1] - 1 - (li_State // self.gg_Grid_Size[0])
            li_Col = li_State % self.gg_Grid_Size[0]
            self.gg_Report['Policy'].append((li_Col, li_Row, self.gg_Agent.Policy(ll_State)))
    
    def Plot(self, Save_Path=None):
        """
        Plot the policy visualization.
        """
        ll_Figure, ll_Axis = plt.subplots(figsize=(6, 6))
        ll_Axis.set_xlim(0, self.gg_Grid_Size[0])
        ll_Axis.set_ylim(0, self.gg_Grid_Size[1])
        ll_Axis.set_xticks(np.arange(0, self.gg_Grid_Size[0]+1, 1))
        ll_Axis.set_yticks(np.arange(0, self.gg_Grid_Size[1]+1, 1))
        ll_Axis.set_xticklabels([]) # Hide x tick labels
        ll_Axis.set_yticklabels([]) # Hide y tick labels
        ll_Axis.set_title("Policy Visualization")

        # Display image as background
        self.gg_Environment.reset()
        ll_Axis.imshow(self.gg_Environment.render(), extent=[0,  self.gg_Grid_Size[0], 0,  self.gg_Grid_Size[1]], zorder=0)

        larr_Directions = {
            self.garr_Action_Order[0]: (0, -1),  # LEFT
            self.garr_Action_Order[1]: (1, 0),   # DOWN
            self.garr_Action_Order[2]: (0, 1),   # RIGHT
            self.garr_Action_Order[3]: (-1, 0)   # UP
        }

        for (li_Col, li_Row, larr_Probabilities) in self.gg_Report['Policy']:
            for li_Action, lf_Probability in enumerate(larr_Probabilities):
                if lf_Probability > 0:
                    li_DX, li_DY = larr_Directions[li_Action]


                    ll_Axis.arrow(
                        li_Col + 0.5, li_Row + 0.5,
                        self.gf_Arrow_Offset * li_DY * (lf_Probability * (1 - self.gf_Arrow_Offset) + self.gf_Arrow_Offset), -self.gf_Arrow_Offset * li_DX * (lf_Probability * (1- self.gf_Arrow_Offset) + self.gf_Arrow_Offset),  # scale and flip y-axis
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6
                        )

                    ll_Axis.text(
                        li_Col + 0.5 + 0.35 * li_DY * (lf_Probability * 0.7 + 0.3),
                        li_Row + 0.5 - 0.35 * li_DX * (lf_Probability * 0.7 + 0.3),
                        f"{lf_Probability:.2f}",
                        color='black',
                        fontsize=7,
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.6)
                    )

        ll_Figure.tight_layout()
        if Save_Path is not None:
            plt.savefig(Save_Path)
        else:
            plt.show()

    
    def Print(self):
        """
        Generate a report of the policy visualization results.
        """
        larr_Report = ["Policy Visualization Report:"]
        larr_Report.append(f"Environment: {self.gg_Environment.spec.id}")
        larr_Report.append(f"Grid Size: {self.gg_Grid_Size[0]} x {self.gg_Grid_Size[1]}")
        larr_Report.append("Policy (State: (Col, Row) -> Action Distribution):")
        for (col, row, action) in self.gg_Report['Policy']:
            larr_Report.append(f" - ({col}, {row}) -> {action}")
        print(Format(larr_Report))