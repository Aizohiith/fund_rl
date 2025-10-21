from fund_rl.agents.base import TAgent


class TRandom_Agent(TAgent):
    def __init__(self, Environment):
        super().__init__(Environment=Environment)
        self.gg_Environment = Environment
        self.Name = "Random"

    def Choose_Action(self, pp_State):
        return self.gg_Environment.action_space.sample()

    def Update(self, pp_Transition):
        pass

    def Policy(self, pi_State):
        lf_Probability = 1.0 / self.Action_Space
        return [lf_Probability] * self.Action_Space