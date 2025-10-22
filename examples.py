def Show_Available_Environments():
    """
    Display a list of all available environments in Gymnasium.
    Note: Agents can only be used in environments with discrete action spaces.
    """
    import gymnasium as gym
    from fund_rl.utility.format import Format

    Print = []
    for Environment_Name in gym.envs.registry.keys():
        Environment_Name = Environment_Name.partition("-")[0]
        if Environment_Name not in Print:
            Print.append(Environment_Name)
    Print.sort()
    Print.insert(0, "Available Environments in Gymnasium:")
    print(Format(Print))

def REINFORCE():
    """
    Demonstration of the REINFORCE agent on the CartPole-v1 environment.
    """
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.local import TLocal_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.utility.run import Run_Agent

    from fund_rl.agents.reinforce import TReinforce_Agent

    print("REINFORCE Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TReinforce_Agent(Environment)

    # Initialize the Tracker
    Tracker = TLocal_Tracker()
    Tracker.Add_Property("Loss")
    Tracker.Add_Property("Entropy")

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=1_000)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Run a few episodes with rendering
    Environment = Make_Environment(Environment_Name, render_mode="human")
    Run_Agent(Environment, Agent, Episodes=5)

def Q_Learning():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.local import TLocal_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.utility.run import Run_Agent

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent

    print("Q-Learning [Epsilon Greedy] Demo")

    # Create the environment
    Environment_Name = "Taxi-v3"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TQ_Epsilon_Agent(Environment)

    # Initialize the Tracker
    Tracker = TLocal_Tracker()

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=2_000, Early_Stopping=5.00)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Run a few episodes with rendering
    Environment = Make_Environment(Environment_Name, render_mode="human")
    Run_Agent(Environment, Agent, Episodes=5)

def DDQN():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.local import TLocal_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.utility.run import Run_Agent

    from fund_rl.agents.ddqn import TDDQN_Agent

    print("DQN Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TDDQN_Agent(Environment, Learning_Rate=0.001, Discount_Factor=0.99, Update_Frequency=1, Swap_Frequency=512, Batch_Size=128)
    Agent.Calculate_Decay_Factor(200)

    # Initialize the Tracker
    Tracker = TLocal_Tracker()
    Tracker.Add_Property("Exploration_Rate")

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=200)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Run a few episodes with rendering
    Environment = Make_Environment(Environment_Name, render_mode="human")
    Run_Agent(Environment, Agent, Episodes=5)

def PPO():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.local import TLocal_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.utility.run import Run_Agent
    import fund_rl.utility.settings as Settings

    from fund_rl.agents.ppo import TPPO_Agent

    print("PPO Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TPPO_Agent(Environment, **Settings.AGENT_PPO_CARTPOLE)

    # Initialize the Tracker
    Tracker = TLocal_Tracker()
    Tracker.Add_Property("Entropy")

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=200)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Run a few episodes with rendering
    Environment = Make_Environment(Environment_Name, render_mode="human")
    Run_Agent(Environment, Agent, Episodes=5)

def WandB_Sweep():
    """
    Demonstration of a Weights & Biases (wandb) hyperparameter sweep using the DDQN agent on the CartPole-v1 environment.
    Note: Ensure you have wandb installed and configured with your account details.
    """
    from fund_rl.sweeps.wb import WandB_Sweep
    from fund_rl.utility.wrappers import Make_Environment
    import fund_rl.environments.snake
    from fund_rl.agents.ppo import TPPO_Agent
    import fund_rl.utility.settings as Settings

    print("WandB Sweep Demo")

    # Create the environment
    Environment = Make_Environment("Snake-v0")

    # Define the sweep configuration
    Config = Settings.SWEEP_PPO

    # Run the sweep
    WandB_Sweep(Environment, TPPO_Agent, "Masters", "Aizohiith", Config, Episodes=1000)

def State_Visitation_Analyzer():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent
    from fund_rl.analyzers.state_visitation import TState_Visitation_Analyzer

    print("State Visitation Analyzer Demo")

    # Create the environment
    Environment_Name = "FrozenLake-v1"
    Environment = Make_Environment(Environment_Name, is_slippery=False)

    # Initialize the Agent
    Agent = TQ_Epsilon_Agent(Environment)
    Agent.Calculate_Decay_Factor(20_000, 0.7, 0.1)

    # Initialize the Tracker
    Tracker = TMetric_Tracker()
    Tracker.Add_Property("Exploration_Rate")

    # Initialize the Analyzer
    Analyzer = TState_Visitation_Analyzer(Most_Common_States=16, Tracker=Tracker)

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=2_000, Early_Stopping=1.00)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

    # Reset the Tracker for evaluation
    Tracker = TMetric_Tracker()

    # Reinitialize the Analyzer
    Analyzer = TState_Visitation_Analyzer(Most_Common_States=16, Tracker=Tracker)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Analyze the state visitation after evaluation
    Analyzer.Analyze()

    # Print the analysis report after evaluation
    Analyzer.Print()

    # Plot the results after evaluation
    Analyzer.Plot()

def Performance_Analyzer():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent

    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.analyzers.preformance import TPerformance_Analyzer
    from fund_rl.utility import settings as Settings

    print("Performance Analyzer Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TPPO_Agent(Environment, **Settings.AGENT_PPO_CARTPOLE)

    # Initialize the Tracker
    Tracker = TMetric_Tracker()

    # Initialize the Analyzer
    Analyzer = TPerformance_Analyzer(Tracker=Tracker)

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=200, Early_Stopping=500.00)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

    # Reset the Tracker for evaluation
    Tracker = TMetric_Tracker()

    # Reinitialize the Analyzer
    Analyzer = TPerformance_Analyzer(Tracker=Tracker)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Analyze the state visitation after evaluation
    Analyzer.Analyze()

    # Print the analysis report after evaluation
    Analyzer.Print()

    # Plot the results after evaluation
    Analyzer.Plot()

def Feature_Distribution_Heatmap_Analyzer():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent

    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.utility.settings import AGENT_DDQN_CARTPOLE
    from fund_rl.utility.settings import ANALYZER_FEATURE_DISTRIBUTION_HEATMAP_CARTPOLE
    from fund_rl.analyzers.feature_distribution_heatmap import TFeature_Distribution_Heatmap_Analyzer

    print("Feature Distribution Heatmap Analyzer Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TDDQN_Agent(Environment, **AGENT_DDQN_CARTPOLE)

    # Initialize the Tracker
    Tracker = TMetric_Tracker()

    # Initialize the Analyzer
    Analyzer = TFeature_Distribution_Heatmap_Analyzer(Tracker=Tracker, **ANALYZER_FEATURE_DISTRIBUTION_HEATMAP_CARTPOLE)

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=200, Early_Stopping=450)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

    # Reset the Tracker for evaluation
    Tracker = TMetric_Tracker()

    # Reinitialize the Analyzer
    Analyzer = TFeature_Distribution_Heatmap_Analyzer(Tracker=Tracker, **ANALYZER_FEATURE_DISTRIBUTION_HEATMAP_CARTPOLE)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Analyze the state visitation after evaluation
    Analyzer.Analyze()

    # Print the analysis report after evaluation
    Analyzer.Print()

    # Plot the results after evaluation
    Analyzer.Plot()

def Decision_Boundary_Analyzer():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent

    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.utility.settings import AGENT_PPO_CARTPOLE
    from fund_rl.utility.settings import ANALYZER_DECISION_BOUNDARY_CARTPOLE
    from fund_rl.analyzers.decision_boundary import TDecision_Boundary_Analyzer

    print("Decision Boundary Analyzer Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TPPO_Agent(Environment, **AGENT_PPO_CARTPOLE)
    # Initialize the Tracker
    Tracker = TMetric_Tracker()

    # Initialize the Analyzer
    Analyzer = TDecision_Boundary_Analyzer(Environment=Environment, Agent=Agent, Tracker=Tracker, **ANALYZER_DECISION_BOUNDARY_CARTPOLE)

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=500, Early_Stopping=450)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

def Consistency_Analyzer():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker

    from fund_rl.agents.ddqn import TDDQN_Agent
    import fund_rl.utility.settings as Settings
    from fund_rl.analyzers.consistency import TConsistency_Analyzer

    print("Consistency Analyzer Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name)

    # Initialize the Tracker
    Tracker = TMetric_Tracker()

    # Initialize the Analyzer
    Analyzer = TConsistency_Analyzer(Environment=Environment, Agent_Class=TDDQN_Agent, Config=Settings.AGENT_DDQN_CARTPOLE, Tracker=Tracker, Episodes=200, Iterations=100)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

def Policy_Visualization_Analyzer():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.metric import TMetric_Tracker
    from fund_rl.training.trainer import Train_Agent

    from fund_rl.agents.ppo import TPPO_Agent
    from fund_rl.utility.settings import AGENT_PPO_FROZENLAKE
    from fund_rl.utility.settings import ANALYZER_POLICY_VISUALISATION_FROZENLAKE
    from fund_rl.analyzers.policy_visualisation import TPolicy_Visualisation_Analyzer
    from fund_rl.utility.wrappers import TOne_Hot_Encode

    print("Policy Visualization Analyzer Demo")

    # Create the environment
    Environment_Name = "FrozenLake-v1"
    Environment = TOne_Hot_Encode(Make_Environment(Environment_Name, is_slippery=False))

    # Initialize the Agent
    Agent = TPPO_Agent(Environment, **AGENT_PPO_FROZENLAKE)

    # Initialize the Tracker
    Tracker = TMetric_Tracker()

    # Initialize the Analyzer
    Analyzer = TPolicy_Visualisation_Analyzer(Environment=(Make_Environment(Environment_Name, render_mode="rgb_array")), One_Hot=True, Agent=Agent, Tracker=Tracker, **ANALYZER_POLICY_VISUALISATION_FROZENLAKE)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=2000, Early_Stopping=0.5)

    # Analyze the state visitation heatmap
    Analyzer.Analyze()

    # Print the analysis report
    Analyzer.Print()

    # Plot the results
    Analyzer.Plot()

def Generating_Videos():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.base import TTracker
    from fund_rl.training.trainer import Train_Agent

    from fund_rl.utility.settings import AGENT_DDQN_CARTPOLE
    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.utility.wrappers import TRecorder

    print("Generating Videos with DQN Demo")

    # Create the environment
    Environment_Name = "CartPole-v1"
    Environment = Make_Environment(Environment_Name, render_mode="rgb_array")

    # Initialize the Agent
    Agent = TDDQN_Agent(Environment, **AGENT_DDQN_CARTPOLE)
    Agent.Calculate_Decay_Factor(200)

    # Initialize the Recorder
    Environment = TRecorder(Environment, Agent, Folder_Name="Videos", Record_Every_N=10)

    # Initialize the Tracker
    Tracker = TTracker()

    # Train the Agent and generate videos during training for every 10th episode
    Train_Agent(Environment, Agent, Tracker, Episodes=200)

def Custom_Environment():
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.trackers.local import TLocal_Tracker
    from fund_rl.training.trainer import Train_Agent
    from fund_rl.evaluate.evaluation import Evaluate_Agent
    from fund_rl.utility.run import Run_Agent
    import fund_rl.environments.snake

    from fund_rl.agents.qlearning import TQ_Epsilon_Agent

    print("Custom Environment Demo")

    # Create the environment
    Environment_Name = "Snake-v0"  # Replace with your custom environment name
    Environment = Make_Environment(Environment_Name)

    # Initialize the Agent
    Agent = TQ_Epsilon_Agent(Environment)

    # Initialize the Tracker
    Tracker = TLocal_Tracker()

    # Train the Agent
    Train_Agent(Environment, Agent, Tracker, Episodes=2_000, Early_Stopping=5.00)

    # Disable training mode for evaluation
    Agent.Is_Training = False

    # Evaluate the Agent
    Evaluate_Agent(Environment, Agent, Tracker, Episodes=100)

    # Run a few episodes with rendering
    Environment = Make_Environment(Environment_Name, render_mode="human")
    Run_Agent(Environment, Agent, Episodes=5)

if __name__ == "__main__":
    #Show_Available_Environments()

    #Agents: 
    #REINFORCE()
    Q_Learning()
    #DDQN()
    #PPO()

    #Analyzers:
    #State_Visitation_Analyzer()
    #Performance_Analyzer()
    #Feature_Distribution_Heatmap_Analyzer()
    #Decision_Boundary_Analyzer()
    #Consistency_Analyzer()
    #Policy_Visualization_Analyzer()

    #Misc:
    #Custom_Environment()
    #WandB_Sweep()
    #Generating_Videos()