# Configuration settings for analyzers
ANALYZER_FEATURE_DISTRIBUTION_HEATMAP_CARTPOLE = {
    "Feature_Names": ["Position", "Velocity", "Angle", "Angular Velocity"],
    "Bins": 50,
    "Feature_Indices": range(4),
}

ANALYZER_POLICY_VISUALISATION_FROZENLAKE = {
    "Action_Order": [0, 1, 2, 3],  # LEFT, DOWN, RIGHT, UP
    "Grid_Size": (4, 4),
}

ANALYZER_POLICY_VISUALISATION_CLIFFWALKING = {
    "Action_Order": [3, 2, 1, 0],  # UP, RIGHT, DOWN, LEFT
    "Grid_Size": (12, 4),
}

ANALYZER_DECISION_BOUNDARY_CARTPOLE = {
    "Feature_Names": ["Position", "Velocity", "Angle", "Angular Velocity"],
    "Feature_Indices": range(4),
    "Feature_Minimum": [-10.0, -10.0, -10.0, -10.0],
    "Feature_Maximum": [10.0, 10.0, 10.0, 10.0],
}

ANALYZER_STATE_VISITATION_LUNAR_LANDER = {
    "parr_Feature_Names": ["Position X", "Position Y", "Angle", "Angular Velocity"],
    "pi_Decimal_Places": 1,
    "pi_Top_N": 50,
    "parr_Feature_Indices": [0, 1, 4, 5],
}
#
# Configuration settings for DDQN agent
    
AGENT_DDQN_CARTPOLE = { #200 Episodes
    "Hidden_Dimensions": 264,
    "Hidden_Layers": 3,
    "Batch_Size": 152,
    "Swap_Frequency": 12,
    "Update_Frequency": 1,
    "Exploration_Decay": 0.2137904205126695,
    "Learning_Rate": 0.0002222757658205644,
}

AGENT_DDQN_SNAKE = { #1_000 Episodes
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Batch_Size": 512,
    "Swap_Frequency": 8,
    "Update_Frequency": 32,
    "Exploration_Decay": 0.9999342140184785,
}
#
# Configuration settings for PPO agent

AGENT_PPO_FROZENLAKE = {
    "Hidden_Dimensions": 8,
    "Hidden_Layers": 3,
    "Learning_Rate": 0.01081653664182426,
    "Iterations": 8,
    "Batch_Size": 168,
    "Minibatch_Size": 56,
    "Entropy_Factor": 0.1,
}

AGENT_PPO_CARTPOLE = { #200 Episodes
    "Hidden_Dimensions": 64,
    "Hidden_Layers": 2,
    "Learning_Rate": 0.0004706893404237208,
    "Iterations": 16,
    "Batch_Size": 856,
    "Minibatch_Size": 24,
}

AGENT_PPO_LUNAR_LANDER = { #1_000 Episodes
    "Hidden_Dimensions": 296,
    "Hidden_Layers": 5,
    "Learning_Rate": 0.00007537747866268359,
    "Iterations": 4,
    "Batch_Size": 1336,
    "Minibatch_Size": 64,
}

AGENT_PPO_SNAKE = {
    "Hidden_Dimensions": 256,
    "Hidden_Layers": 3,
    "Learning_Rate": 0.000293264649968785,
    "Iterations": 16,
    "Batch_Size": 480,
    "Entropy_Factor": 0.006432220592416948,
    "Minibatch_Size": 120,
}
#
# Configuration settings for REINFORCE agent

AGENT_REINFORCE_CARTPOLE = { #200 Episodes
    "Hidden_Dimensions": 8,
    "Hidden_Layers": 3,
    "Learning_Rate": 0.0067272261749594655,
}

AGENT_REINFORCE_FROZENLAKE = {
    "Hidden_Dimensions": 8,
    "Hidden_Layers": 3,
    "Learning_Rate": 0.01081653664182426,
}

AGENT_REINFORCE_LUNAR_LANDER = { #10_000 Episodes
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Learning_Rate": 1e-4,
}
#
# Configuration settings for WandB sweeps

SWEEP_REINFORCE = {
    "method": "bayes",
    "metric": {"name": "Filtered_Reward", "goal": "maximize"},
    "parameters": {
        "Learning_Rate": {
            "distribution": "log_uniform_values",
            "min": 1e-12,
            "max": 1.0,
        },
        "Hidden_Layers": {
            "values": [1, 2, 3, 4, 5]
        },
        "Hidden_Dimensions": {
            "distribution": "q_log_uniform_values",
            "max": 512,
            "min": 8,
            "q": 8,
        },
    }
}

SWEEP_PPO = {
    "method": "bayes",
    "metric": {"name": "Filtered_Reward", "goal": "maximize"},
    "parameters": {
        "Learning_Rate": {
            "distribution": "log_uniform_values",
            "min": 1e-12,
            "max": 1.0,
        },
        "Entropy_Factor": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "Hidden_Layers": {
            "values": [1, 2, 3, 4, 5]
        },
        "Hidden_Dimensions": {
            "distribution": "q_log_uniform_values",
            "max": 512,
            "min": 8,
            "q": 8,
        },
        "Iterations": {
            "values": [1, 2, 4, 8, 16]
        },
        "Batch_Size": {
            "distribution": "q_log_uniform_values",
            "max": 2048,
            "min": 8,
            "q": 8,
        },
        "Minibatch_Size": {
            "distribution": "q_log_uniform_values",
            "max": 512,
            "min": 8,
            "q": 8,
        }
    }
}

SWEEP_DDQN = {
    "method": "bayes",
    "metric": {"name": "Filtered_Reward", "goal": "maximize"},
    "parameters": {
        "Learning_Rate": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 1.0,
        },
        "Hidden_Layers": {
            "values": [1, 2, 3, 4, 5]
        },
        "Hidden_Dimensions": {
            "distribution": "q_log_uniform_values",
            "max": 512,
            "min": 8,
            "q": 8,
        },
        "Swap_Frequency": {
            "distribution": "q_log_uniform_values",
            "max": 4096,
            "min": 2,
            "q": 2,
        },
        "Batch_Size": {
            "distribution": "q_log_uniform_values",
            "max": 2048,
            "min": 8,
            "q": 8,
        },
        "Update_Frequency": {
            "distribution": "q_log_uniform_values",
            "max": 512,
            "min": 1,
            "q": 1,
        },
        "Exploration_Decay": {
            "distribution": "uniform",
            "min": 0.001,
            "max": 1.0,
        }
    }
}
#