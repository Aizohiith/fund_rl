def Exploration_vs_Exploitation():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.utility.run import Run_Agent

    import matplotlib.pyplot as plt

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.agents.qlearning import TQ_UCB_Agent

    from fund_rl.environments.multi_armed_bandit import TMulti_Armed_Bandit_Environment
    from fund_rl.utility.filters import Parrallel_Average_Filter
    from fund_rl.utility.filters import EMA_Filter
    from tqdm import tqdm

    Rewards = {'Epsilon-Greedy-Decay': [], 'Greedy': [], 'Optimistic': [], 'UCB': [], 'Epsilon-Greedy': []}

    for C0 in tqdm(range(1000)):

        Environment = Make_Environment("multi-armed-bandit-v0", n_arms=100)

        Agents = []

        Agents.append(TQ_Epsilon_Agent(Environment, Exploration_Rate=1.0))
        Agents[0].Calculate_Decay_Factor(5000, 0.7, 0.1)
        Agents[0].Name = "Epsilon-Greedy-Decay"
        Agents.append(TQ_Epsilon_Agent(Environment, Exploration_Rate=0.0, Min_Exploration_Rate=0.0))
        Agents[1].Name = "Greedy"
        Agents.append(TQ_Epsilon_Agent(Environment, Initial_Q_Value=1.0, Exploration_Rate=0.0, Min_Exploration_Rate=0.0))
        Agents[2].Name = "Optimistic"
        Agents.append(TQ_UCB_Agent(Environment, C=2))
        Agents[3].Name = "UCB"
        Agents.append(TQ_Epsilon_Agent(Environment, Exploration_Rate=0.1, Min_Exploration_Rate=0.1, Exploration_Decay=0.0))
        Agents[4].Name = "Epsilon-Greedy"


        Trackers = [TMetric_Tracker(Show_Progress=False) for _ in range(len(Agents))]
        for E1 in Trackers:
            E1.Add_Property("Filtered_Reward")

        for C1 in range(len(Agents)):
            Train_Agent(Environment, Agents[C1], Trackers[C1], Episodes=5000)

        for C1 in range(len(Agents)):
            Rewards[Agents[C1].Name].append(Trackers[C1].Data("Reward"))

    for Key in Rewards.keys():
        Reward = Parrallel_Average_Filter(Rewards[Key])
        Reward_Filtered = EMA_Filter(Reward, 0.9)
        Line, = plt.plot(Reward_Filtered, label=Key)
        plt.plot(Reward, alpha=0.3, color=Line.get_color())
    plt.title("Exploration vs Exploitation")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()

def Implementation_Validation():
    """
    Validate the implementation of various agents on a simple environment. FrozenLake
    Goal here is to ensure that all agents can learn to solve the environment.
    not to compare their performance but to ensure that they all are implemented correctly.
    """
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.agents.reinforce import TReinforce_Agent
    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.agents.random import TRandom_Agent
    from fund_rl.utility.wrappers import TOne_Hot_Encode
    import matplotlib.pyplot as plt

    import numpy as np

    import fund_rl.utility.settings as Settings

    # Random Agent Action Mask to prevent invalid moves
    def Random_Action_Selection_Mask(State):
        State = np.argmax(State)

        Action_Mask = np.ones(4)

        # 0 : LEFT, 1 : DOWN, 2: RIGHT, 3: UP
        if State in range(0 , 4):
            Action_Mask[3] = 0.0  # Up
        if State in range(12 , 16):
            Action_Mask[1] = 0.0  # Down
        if State % 4 == 0:
            Action_Mask[0] = 0.0  # Left
        if State % 4 == 3:
            Action_Mask[2] = 0.0  # Right

        return Action_Mask

    # Create Environment
    Environment_Name = "FrozenLake-v1"
    Environment = TOne_Hot_Encode(Make_Environment(Environment_Name, is_slippery=False))

    # Setup Agents and Trackers
    Agents_Tracker = []
    Agents_Tracker.append((TQ_Epsilon_Agent(Environment), TMetric_Tracker()))
    Agents_Tracker.append((TReinforce_Agent(Environment, **Settings.AGENT_REINFORCE_FROZENLAKE), TMetric_Tracker()))
    Agents_Tracker.append((TDDQN_Agent(Environment, **Settings.AGENT_DDQN_FROZENLAKE), TMetric_Tracker()))
    Agents_Tracker.append((TPPO_Agent(Environment, **Settings.AGENT_PPO_FROZENLAKE), TMetric_Tracker()))
    Agents_Tracker.append((TRandom_Agent(Environment, Action_Mask_Function=Random_Action_Selection_Mask), TMetric_Tracker()))
    

    # For each agent
    for Agent, Tracker  in Agents_Tracker:

        # Add properties to tracker
        Tracker.Add_Property("Filtered_Reward")

        # Make a fresh environment
        Environment = TOne_Hot_Encode(Make_Environment(Environment_Name, is_slippery=False))

        # Train the Agent
        Train_Agent(Environment, Agent, Tracker, Episodes=2_000, Early_Stopping=1.0)

    # Plot the results
    for Agent, Tracker in Agents_Tracker:
        plt.plot(Tracker.Data("Filtered_Reward"), label=Agent.Name)
    plt.title("Agent Benchmark on " + Environment_Name)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()

    """
    All agents showed learning behavior and were able to solve the environment.
    The only exception was the REINFORCE agent which struggled to learn the environment
    due to the sparse reward structure of FrozenLake. This indicates that while the implementation is correct, 
    certain agents may require more sophisticated techniques or architectures to handle specific environments effectively.
    It did however show some learning trends indicating that the implementation is correct.
    """

def Policy_Visualization():
    """
    Visualize the policy of a trained agent in a 2D environment. ForzenLake
    Compare how different agents learn to navigate the environment.
    ne imporatnt thing to note is that the environment is deterministic (is_slippery=False)
    to ensure that the agents can learn a clear policy.
    another is that policy optimizations tends to show smooth policies while value based methods tend to show more discrete policies.
    and rapid changes in policy due to maximizing value.
    """
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.agents.reinforce import TReinforce_Agent
    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.agents.random import TRandom_Agent
    from fund_rl.utility.wrappers import TOne_Hot_Encode
    from fund_rl.analyzers.policy_visualisation import TPolicy_Visualisation_Analyzer
    import matplotlib.pyplot as plt

    import numpy as np

    import fund_rl.utility.settings as Settings

    # Random Agent Action Mask to prevent invalid moves
    def Random_Action_Selection_Mask(State):
        State = np.argmax(State)

        Action_Mask = np.ones(4)

        # 0 : LEFT, 1 : DOWN, 2: RIGHT, 3: UP
        if State in range(0 , 4):
            Action_Mask[3] = 0.0  # Up
        if State in range(12 , 16):
            Action_Mask[1] = 0.0  # Down
        if State % 4 == 0:
            Action_Mask[0] = 0.0  # Left
        if State % 4 == 3:
            Action_Mask[2] = 0.0  # Right


        return Action_Mask

    # Create Environment
    Environment_Name = "FrozenLake-v1"
    Environment = TOne_Hot_Encode(Make_Environment(Environment_Name, is_slippery=False))

    # Setup Agents
    Agents = []
    Agents.append(TQ_Epsilon_Agent(Environment, **Settings.AGENT_Q_LEARNING_FROZENLAKE))
    Agents.append(TReinforce_Agent(Environment, **Settings.AGENT_REINFORCE_FROZENLAKE))
    Agents.append(TDDQN_Agent(Environment, **Settings.AGENT_DDQN_FROZENLAKE))
    Agents.append(TPPO_Agent(Environment, **Settings.AGENT_PPO_FROZENLAKE))
    Agents.append(TRandom_Agent(Environment, Action_Mask_Function=Random_Action_Selection_Mask))

    # For each agent
    for Agent in Agents:

        #Tracker
        Tracker = TMetric_Tracker(Filter_Strength=0.9)
        Tracker.Add_Property("Filtered_Reward")
        
        #Analyzer
        Analyzer = TPolicy_Visualisation_Analyzer(Environment=(Make_Environment(Environment_Name, render_mode="rgb_array")), One_Hot=True, Agent=Agent, Tracker=Tracker, **Settings.ANALYZER_POLICY_VISUALISATION_FROZENLAKE)
        
        # Make a fresh environment
        Environment = TOne_Hot_Encode(Make_Environment(Environment_Name, is_slippery=False))

        # for each 500 episodes
        for C1 in range(0, 2_000, 500):
            # Analyze the policy visualization
            Analyzer.Analyze()

            # Plot the policy visualization
            if C1 == 0:
                Analyzer.Plot(f"experiments/policy/Policy_Visualization_{Agent.Name}_Episodes_{C1}_Reward_{0}.png")
            else:
                Analyzer.Plot(f"experiments/policy/Policy_Visualization_{Agent.Name}_Episodes_{C1}_Reward_{Tracker.Data('Filtered_Reward')[-1]:.2f}.png")
            
            # Train the Agent
            Train_Agent(Environment, Agent, Tracker, Episodes=500, Early_Stopping=(C1+500)/2_000)

        # Final Analysis
        Analyzer.Analyze()

        # Plot the final policy visualization
        Analyzer.Plot(f"experiments/policy/Policy_Visualization_{Agent.Name}_Episodes_{2000}_Reward_{Tracker.Data('Filtered_Reward')[-1]:.2f}.png")

def Learning_Rate_Impact():
    """
    Analyze the impact of learning rate on agent behavior. Snake Performance
    What should be observed is that higher learning rates can lead to faster initial learning but may cause instability and suboptimal performance in the long run.
    Also PPO agents tend to be more sensitive to learning rate changes compared to DDQN agents.
    """

    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    import fund_rl.utility.settings as Settings
    from fund_rl.environments.snake import TSnake_Environment
    import copy
    from fund_rl.analyzers.preformance import TPerformance_Analyzer

    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.agents.ppo import TPPO_Agent

    import matplotlib.pyplot as plt

    # Create Environment
    Environment = Make_Environment("Snake-v0")

    # Setup Agents and Analyzers
    Agents_Analyzer = []
    Agents_Analyzer.append((TDDQN_Agent(Environment, **Settings.AGENT_DDQN_SNAKE), TPerformance_Analyzer()))
    Config = copy.deepcopy(Settings.AGENT_DDQN_SNAKE)
    Config['Learning_Rate'] = Config['Learning_Rate'] * 0.1
    Agents_Analyzer.append((TDDQN_Agent(Environment, **Config), TPerformance_Analyzer()))
    Agents_Analyzer[-1][0].Name += " LR: x0.1"
    Config = copy.deepcopy(Settings.AGENT_DDQN_SNAKE)
    Config['Learning_Rate'] = Config['Learning_Rate'] * 10.0
    Agents_Analyzer.append((TDDQN_Agent(Environment, **Config), TPerformance_Analyzer()))
    Agents_Analyzer[-1][0].Name += " LR: x10.0"

    Agents_Analyzer.append((TPPO_Agent(Environment, **Settings.AGENT_PPO_SNAKE), TPerformance_Analyzer()))
    Config = copy.deepcopy(Settings.AGENT_PPO_SNAKE)
    Config['Learning_Rate'] = Config['Learning_Rate'] * 0.1
    Agents_Analyzer.append((TPPO_Agent(Environment, **Config), TPerformance_Analyzer()))
    Agents_Analyzer[-1][0].Name += " LR: x0.1"
    Config = copy.deepcopy(Settings.AGENT_PPO_SNAKE)
    Config['Learning_Rate'] = Config['Learning_Rate'] * 10.0
    Agents_Analyzer.append((TPPO_Agent(Environment, **Config), TPerformance_Analyzer()))
    Agents_Analyzer[-1][0].Name += " LR: x10.0"

    # For each agent
    for Agent, Analyzer in Agents_Analyzer:

        # Create Tracker
        Tracker = TMetric_Tracker()
        Analyzer.Set_Tracker(Tracker)

        # Train the Agent
        Train_Agent(Environment, Agent, Tracker, Episodes=1_000)

        # Analyze Performance
        Analyzer.Analyze()

    # Plot the results
    plt.clf()
    for Agent, Analyzer in Agents_Analyzer:
        
        plt.subplot(2, 2, 1)
        plt.plot(Analyzer.Report()['Rewards Filtered'], label=Agent.Name)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title("Agent's Rewards")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(Analyzer.Report()['Losses Filtered'], label=Agent.Name)
        plt.xlabel('Episodes')
        plt.ylabel('Losses')
        plt.title("Agent's Losses")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(Analyzer.Report()['Mean Rewards'], label=Agent.Name)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.title("Agent's Mean Rewards")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(Analyzer.Report()['STD Rewards'], label=Agent.Name)
        plt.xlabel('Episodes')
        plt.ylabel('STD Rewards')
        plt.title("Agent's STD Rewards")
        plt.legend()

        plt.tight_layout()


    plt.show()

def Exploration_Consistency():
    """
    Examine how consistent exploration strategies affect learning. Cartpole Consistency Analysis
    Taking PPO and REINFORCE agents and varying their entropy regularization factors to see how it impacts the consistency of their learning outcomes across multiple training runs.
    Higher entropy factors should lead to more exploration and potentially more consistent learning outcomes.
    """
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    import fund_rl.utility.settings as Settings

    from fund_rl.agents.reinforce import TReinforce_Agent
    from fund_rl.agents.ppo import TPPO_Agent

    from fund_rl.analyzers.consistency import TConsistency_Analyzer

    import copy
    import matplotlib.pyplot as plt

    # Create Environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Setup Agents and Configurations
    Agent_Config = []
    Agent_Config.append((TReinforce_Agent, copy.deepcopy(Settings.AGENT_REINFORCE_CARTPOLE)))
    Agent_Config.append((TPPO_Agent, copy.deepcopy(Settings.AGENT_PPO_CARTPOLE)))

    # For each agent
    for E1 in Agent_Config:
        Agent_Class, Config = E1

        # For each entropy factor
        for E2 in [0.0, 1.0, 10.0]:
            # Update the configuration
            Config = copy.deepcopy(E1[1])
            Config["Entropy_Factor"] = Config["Entropy_Factor"] * E2

            # Create Tracker and Analyzer
            Tracker = TMetric_Tracker()
            Analyzer = TConsistency_Analyzer(Environment, Agent_Class, Config, Tracker=Tracker, Episodes=200, Iterations=200)

            # Analyze
            Analyzer.Analyze()

            # Plot the results
            plt.plot(Analyzer.Report()['Mean Rewards'], label='Entropy Factor: ' + str(Config['Entropy_Factor']))
            plt.fill_between(range(len(Analyzer.Report()['Mean Rewards'])), Analyzer.Report()['Mean Rewards'] - Analyzer.Report()['STD Rewards'], Analyzer.Report()['Mean Rewards'] + Analyzer.Report()['STD Rewards'], alpha=0.2)
            Analyzer.Print(f"experiments/consistency/Consistency_{Agent_Class.__name__}_Entropy_{Config['Entropy_Factor']}.txt")
        
        # Finalize Plot
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Consistency Analysis: ' + Agent_Class.__name__)
        plt.legend()
        plt.savefig(f"experiments/consistency/Consistency_{Agent_Class.__name__}.png")
        plt.clf()

def State_Visitation():
    """Analyze the impact of high dimensional one-hot encoding on agent performance. Taxi State Visitation"""
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    import fund_rl.utility.settings as Settings

    from fund_rl.analyzers.state_visitation import TState_Visitation_Analyzer
    from fund_rl.analyzers.feature_distribution_heatmap import TFeature_Distribution_Heatmap_Analyzer
    from fund_rl.evaluate.evaluation import Evaluate_Agent

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.agents.ddqn import TDDQN_Agent

    # State Decoder
    def State_Decoder(State):
        for C1 in range(5):
            for C2 in range(5):
                for C3 in range(5):
                    for C4 in range(4):
                        Result = ((C1 * 5 + C2) * 5 + C3) * 4 + C4
                        if Result == State:
                            return C1, C2, C3, C4

    # Create Environment
    Environment = Make_Environment("Taxi-v3")

    # Create Agent
    Agents = []
    Agents.append(TQ_Epsilon_Agent(Environment))

    # Create Tracker
    Tracker = TMetric_Tracker(Show_Progress=True)

    # Create Analyzer
    Analyzer = TState_Visitation_Analyzer(Most_Common_States=400, Tracker=Tracker)

    # Train Agent
    Train_Agent(Environment, Agents[0], Tracker, Episodes=5_000)

    # Analyze State Visitation
    Analyzer.Analyze()
    Analyzer.Plot("experiments/state_visitation/Taxi_State_Visitation_Training")

    # Decode States
    Analyzer.Report()['Top_States']
    Decoded_States = [State_Decoder(State) for State in Analyzer.Report()['Top_States']]
    
    # Create new tracker
    Tracker = TMetric_Tracker()

    # Create Heatmap Analyzer
    Heatmap_Analyzer = TFeature_Distribution_Heatmap_Analyzer(Feature_Indices=[0,1,2,3], Feature_Names=["Taxi Row", "Taxi Column", "Passenger Location", "Destination"], Tracker=Tracker, Bins=5)
    
    # Log States
    for C1 in range(len(Decoded_States)):
        Tracker.Log("State", Decoded_States[C1], C1)

    # Analyze
    Heatmap_Analyzer.Analyze()

    # Plot Heatmaps
    Heatmap_Analyzer.Plot("experiments/state_visitation/Taxi_State_Visitation_Training")

    # Create Environment
    Environment = Make_Environment("Taxi-v3")

    # Set agent
    Agents[0].Is_Training = False

    # Create  Tracker
    Tracker = TMetric_Tracker()

    # Create Analyzer
    Analyzer = TState_Visitation_Analyzer(Most_Common_States=400, Tracker=Tracker)

    # Evaluate Agent
    Evaluate_Agent(Environment, Agents[0], Tracker, Episodes=5_000)

    # Analyze State Visitation
    Analyzer.Analyze()
    Analyzer.Plot("experiments/state_visitation/Taxi_State_Visitation_Evaluation")
    Analyzer.Report()['Top_States']

    # Decode States
    Decoded_States = [State_Decoder(State) for State in Analyzer.Report()['Top_States']]
    
    # Create new trackers
    Tracker_All = TMetric_Tracker()
    Tracker = TMetric_Tracker()

    # Create Heatmap Analyzers
    Heatmap_Analyzer_All = TFeature_Distribution_Heatmap_Analyzer(Feature_Indices=[0,1,2,3], Feature_Names=["Taxi Row", "Taxi Column", "Passenger Location", "Destination"], Tracker=Tracker_All, Bins=5)
    Heatmap_Analyzer = TFeature_Distribution_Heatmap_Analyzer(Feature_Indices=[0,1,2,3], Feature_Names=["Taxi Row", "Taxi Column", "Passenger Location", "Destination"], Tracker=Tracker, Bins=5)
    
    # Log States
    for C1 in range(len(Decoded_States)):
        if C1 < 100:
            Tracker.Log("State", Decoded_States[C1], C1)
        Tracker_All.Log("State", Decoded_States[C1], C1)

    # Analyze
    Heatmap_Analyzer_All.Analyze()
    Heatmap_Analyzer.Analyze()

    # Plot Heatmaps
    Heatmap_Analyzer_All.Plot("experiments/state_visitation/Taxi_State_Visitation_All")
    Heatmap_Analyzer.Plot("experiments/state_visitation/Taxi_State_Visitation")

def Policy_VS_Value_Learning_Rates():
    """Compare the effects of different learning rates for policy-based and value-based agents. LunarLander Feature Heatmaps"""
    pass
    
def Feature_Selection():
    """Analyze the impact of feature selection on agent performance. CartPole Decision Boundaries"""

    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    import fund_rl.utility.settings as Settings

    from fund_rl.analyzers.decision_boundary import TDecision_Boundary_Analyzer
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.analyzers.preformance import TPerformance_Analyzer
    from fund_rl.utility.wrappers import TSelect_Observations

    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.agents.random import TRandom_Agent

    import numpy as np
    import copy

    # Create Environment
    Environment = Make_Environment("CartPole-v1")

    # Create Agents
    Agents = []
    Agents.append(TDDQN_Agent(Environment, **Settings.AGENT_DDQN_CARTPOLE))
    Agents.append(TPPO_Agent(Environment, **Settings.AGENT_PPO_CARTPOLE))

    # For each agent
    for Agent in Agents:

        # Create  Tracker
        Tracker = TMetric_Tracker(Show_Progress=True)
        Tracker.Add_Property("Filtered_Reward")

        # Create Analyzer and Analyze Pre-Training Decision Boundary
        Analyzer = TDecision_Boundary_Analyzer(Environment, Agent, **Settings.ANALYZER_DECISION_BOUNDARY_CARTPOLE, Tracker=Tracker)
        Analyzer.Analyze()
        Analyzer.Plot("experiments/decision_boundary/Pre_Training_Decision_Boundary_" + Agent.Name)
        
        # Create new analyzer for training
        Analyzer = TDecision_Boundary_Analyzer(Environment, Agent, **Settings.ANALYZER_DECISION_BOUNDARY_CARTPOLE, Tracker=Tracker)
        # Train the Agent
        Train_Agent(Environment, Agent, Tracker, Episodes=200, Early_Stopping=500.0)

        # Analyze Post-Training Decision Boundary
        Analyzer.Analyze()
        Analyzer.Plot("experiments/decision_boundary/Post_Training_Decision_Boundary_" + Agent.Name)

        # Evaluate Agent Performance
        Agent.Is_Training = False
        Tracker = TMetric_Tracker()
        Analyzer = TPerformance_Analyzer(Tracker)
        Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)
        Analyzer.Analyze()
        Analyzer.Plot("experiments/decision_boundary/Performance_CartPole_" + Agent.Name + ".png")

    # Sampling decision boundary with feature selection
    def Action_Selection_Mask(State):
        larr_Action_Scores = np.zeros(Environment.action_space.n)

        for C1 in range(4):
            for C2 in range(4):
                if C1 == C2:
                    continue
                Temp_State = copy.deepcopy(State)
                Temp_State[C1] = 0
                Temp_State[C2] = 0
                PPO_Action = Agents[1].Choose_Action(Temp_State)
                larr_Action_Scores[PPO_Action] += 1.0

        Max_Score = np.max(larr_Action_Scores)
        return np.where(larr_Action_Scores == Max_Score, 1.0, 0.0)

    # Create Random Agent with Action Mask
    Agent = TRandom_Agent(Environment, Action_Mask_Function=Action_Selection_Mask)

    # Evaluate Agent Performance
    Agent.Is_Training = False
    Tracker = TMetric_Tracker()
    Analyzer_Preformance = TPerformance_Analyzer(Tracker)
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)
    Analyzer_Preformance.Analyze()
    Analyzer_Preformance.Plot("experiments/decision_boundary/Performance_CartPole_Random_Feature_Selected.png") 

    # Create Q-Learning Agent with Feature Selection
    Environment = TSelect_Observations(Make_Environment("CartPole-v1"), [0, 1, 2, 3])
    Agent = TQ_Epsilon_Agent(Environment)
    Agent.Calculate_Decay_Factor(100_000, 0.7, 0.01)
    Tracker = TMetric_Tracker()
    Train_Agent(Environment, Agent, Tracker, Episodes=100_000, Early_Stopping=500.0)
    Agent.Is_Training = False
    Tracker = TMetric_Tracker()
    Analyzer_Preformance = TPerformance_Analyzer(Tracker)
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)
    Analyzer_Preformance.Analyze()
    Analyzer_Preformance.Plot("experiments/decision_boundary/Performance_CartPole_Feature_Selected_All.png")

    # Create Q-Learning Agent with Feature Selection
    Environment = TSelect_Observations(Make_Environment("CartPole-v1"), [1, 2, 3])
    Agent = TQ_Epsilon_Agent(Environment)
    Agent.Calculate_Decay_Factor(100_000, 0.7, 0.01)
    Tracker = TMetric_Tracker()
    Train_Agent(Environment, Agent, Tracker, Episodes=100_000, Early_Stopping=500.0)
    Agent.Is_Training = False
    Tracker = TMetric_Tracker()
    Analyzer_Preformance = TPerformance_Analyzer(Tracker)
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)
    Analyzer_Preformance.Analyze()
    Analyzer_Preformance.Plot("experiments/decision_boundary/Performance_CartPole_Feature_Selected.png")

def Decision_Boundary_Linear():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    import fund_rl.utility.settings as Settings

    from fund_rl.analyzers.decision_boundary import TDecision_Boundary_Analyzer
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.analyzers.preformance import TPerformance_Analyzer
    from fund_rl.utility.wrappers import TSelect_Observations

    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.agents.random import TRandom_Agent

    import numpy as np
    import copy

    Environment = Make_Environment("CartPole-v1")


    def Action_Selection_Mask(State):
        # Always allow all actions
        larr_Action_Scores = np.zeros(Environment.action_space.n)

        BLUE = 0
        RED = 1

        x = State[3] / -3
        if (State[2] < x):
            larr_Action_Scores[BLUE] += 1.0
        else:
            larr_Action_Scores[RED] += 1.0

        if (State[2] > 0):
            larr_Action_Scores[RED] += 1.0
        else:
            larr_Action_Scores[BLUE] += 1.0

        y = -0.25 * State[0]

        if (State[3] < y):
            larr_Action_Scores[BLUE] += 1.0
        else:
            larr_Action_Scores[RED] += 1.0

        Y = -0.5 * State[0]

        if (State[1] < Y):
            larr_Action_Scores[BLUE] += 1.0
        else:
            larr_Action_Scores[RED] += 1.0

        Y = -0.25 * State[1]

        if (State[2] < Y):
            larr_Action_Scores[BLUE] += 1.0
        else:
            larr_Action_Scores[RED] += 1.0

        Y = -0.75 * State[1]

        if (State[3] < Y):
            larr_Action_Scores[BLUE] += 1.0
        else:
            larr_Action_Scores[RED] += 1.0

        Max_Score = np.max(larr_Action_Scores)
        return np.where(larr_Action_Scores == Max_Score, 1.0, 0.0)


    Agent = TRandom_Agent(Environment, Action_Mask_Function=Action_Selection_Mask)

    Agent.Is_Training = False
    Tracker = TMetric_Tracker()
    Analyzer_Preformance = TPerformance_Analyzer(Tracker)
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    Analyzer_Preformance.Analyze()
    Analyzer_Preformance.Plot("experiments/decision_boundary/Performance_CartPole_ymxc.png")


if __name__ == "__main__":
    #Exploration_vs_Exploitation()
    #Implementation_Validation()
    #Policy_Visualization()
    #Learning_Rate_Impact()
    #Exploration_Consistency()
    #State_Visitation()
    #Feature_Selection()
    Decision_Boundary_Linear()
