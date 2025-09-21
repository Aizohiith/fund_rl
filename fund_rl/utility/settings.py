ANALYZER_STATE_VISITATION_CARTPOLE = {
    "parr_Feature_Names": ["Position", "Velocity", "Angle", "Angular Velocity"],
    "pi_Decimal_Places": 1,
    "pi_Top_N": 50,
    "parr_Feature_Indices": range(4),
}

ANALYZER_STATE_VISITATION_LUNAR_LANDER = {
    "parr_Feature_Names": ["Position X", "Position Y", "Angle", "Angular Velocity"],
    "pi_Decimal_Places": 1,
    "pi_Top_N": 50,
    "parr_Feature_Indices": [0, 1, 4, 5],
}

# AGENT_DDQN_CARTPOLE = { #200 Episodes
#     "pi_Hidden_Dimensions": 128,
#     "pi_Hidden_Layers": 1,
#     "pi_Batch_Size": 2048,
#     "pi_Swap_Frequency": 4,
#     "pi_Update_Frequency": 4,
# }

AGENT_DDQN_CARTPOLE = { #200 Episodes
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Batch_Size": 64,
    "Swap_Frequency": 100,
    "Update_Frequency": 4,
    "Exploration_Decay": 0.995,
    "Learning_Rate": 0.001,
}

AGENT_DDQN_BREAKOUT_RAM = {
    "Hidden_Dimensions": 256,           # 512 works but is overkill for 128-dim RAM input
    "Hidden_Layers": 2,                 # Two layers give enough non-linearity
    "Batch_Size": 64,                   # You can afford a bigger batch for faster learning on small inputs
    "Swap_Frequency": 5000,             # Faster swap improves stability in low-dim environments
    "Update_Frequency": 4,              # Standard update frequency
    "Exploration_Decay": 0.9999,         # Slightly faster decay due to faster convergence on RAM input
    "Learning_Rate": 0.0005,            # Slightly higher learning rate can help RAM-based learning
}


AGENT_DDQN_SNAKE = { #1_000 Episodes
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Batch_Size": 512,
    "Swap_Frequency": 8,
    "Update_Frequency": 32,
    "Exploration_Decay": 0.9999342140184785,
}

AGENT_DDQN_PONG_RAM = { #1_000 Episodes
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Batch_Size": 2048,
    "Swap_Frequency": 8,
    "Update_Frequency": 4,
    "Exploration_Decay": 0.9999934212070889,
}

AGENT_PPO_CARTPOLE = { #200 Episodes
    "Hidden_Dimensions": 64,
    "Hidden_Layers": 2,
    "Learning_Rate": 0.0003,
    "Iterations": 8,
}

AGENT_PPO_SNAKE = {
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Learning_Rate": 1e-3,
    "Iterations": 4,
    "Batch_Size": 2048,
    "Minibatch_Size": 512,
}

AGENT_PPO_PONG_RAM = {
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Learning_Rate": 5e-4,
    "Iterations": 4,
    "Batch_Size": 512,
    "Minibatch_Size": 128,
}

AGENT_REINFORCE_CARTPOLE = { #200 Episodes
    "Hidden_Dimensions": 8,
    "Hidden_Layers": 3,
    "Learning_Rate": 0.0067272261749594655,
}

AGENT_REINFORCE_LUNAR_LANDER = { #10_000 Episodes
    "Hidden_Dimensions": 128,
    "Hidden_Layers": 1,
    "Learning_Rate": 1e-4,
}


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