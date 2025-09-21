
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

def WandB_Sweep():
    """
    Demonstration of a Weights & Biases (wandb) hyperparameter sweep using the DDQN agent on the CartPole-v1 environment.
    Note: Ensure you have wandb installed and configured with your account details.
    """
    from fund_rl.sweeps.wb import WandB_Sweep
    from fund_rl.utility.wrappers import Make_Environment
    from fund_rl.agents.ddqn import TDDQN_Agent
    from fund_rl.utility.settings import SWEEP_DDQN

    print("WandB Sweep Demo")

    # Create the environment
    Environment = Make_Environment("CartPole-v1")

    # Define the sweep configuration
    Config = SWEEP_DDQN

    # Run the sweep
    WandB_Sweep(Environment, TDDQN_Agent, "Masters", "Aizohiith", Config, Episodes=200)

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

if __name__ == "__main__":
    #Show_Available_Environments()
    #REINFORCE()
    Q_Learning()
    #DDQN()
    #WandB_Sweep()
    #Generating_Videos()