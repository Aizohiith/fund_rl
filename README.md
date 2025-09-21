# fund_rl 🧠

**`fund_rl`** is a custom reinforcement learning library designed for **educational research** and **interpretability**. Unlike traditional RL libraries optimized for performance, `fund_rl` emphasizes **transparency**, **simplicity**, and **understandability** of agent behavior.

This makes it a perfect tool for students, researchers, and anyone looking to **learn or teach the core ideas of RL** through visual, traceable implementations and modular agent design.

---

### 🚧 In Development

> ⚠️ **This library is still under active development** and not yet ready for production use.

The core functionality is present, but several components are incomplete or undergoing refinement to improve consistency, readability, and overall usability.

#### 🔧 Known Areas That Need Work

- ❗ **Inconsistent comments** across modules and functions
- 📝 **Variable and function renaming** for clarity and naming conventions
- 📚 **Lack of docstrings** and inline documentation in agent implementations
- 🧪 **Analyzer tools are missing** or partially implemented
- 🧱 **Some utility functions** are still placeholders or not yet implemented
- 📉 **Certain agents (e.g., PPO)** need modular cleanup and restructuring

#### 🧠 Project Philosophy

> `fund_rl` is designed for **clarity over optimization**.

Unlike high-performance RL libraries, `fund_rl` intentionally prioritizes:

- Transparent agent implementations
- 1:1 flow diagram to code mapping
- Code that favors learning and explainability over performance

Contributions, testing, and suggestions are very welcome as the library continues to mature.

---

## 🧩 Key Features

- Multiple agent implementations: `Q-Learning`, `REINFORCE`, `DDQN`, `PPO`
- Built-in rollout buffer
- Modular support for:
  - Action selection strategies
  - Neural networks (PyTorch-based)
  - Custom environments
  - Evaluation and training metrics
- Easy-to-read code structure aligned with flow diagrams
- Lightweight tracing, logging, and visualization tools
- Experiment configuration via sweep support
- Support for recording training sessions as video demos

---

## 📁 Code Structure Overview

Each agent inherits from a common base class `TAgent` and is structured to reflect a **1:1 mapping** with flow diagrams included in the original research.

**Typical agent functions:**
- `__init__()`: Initializes the agent
- `choose_action(state: np.ndarray) -> int`: Chooses an action
- `update(transition)`: Updates the agent based on experience
- `policy(state: np.ndarray) -> np.ndarray`: Outputs action probabilities
- Optional: `save()`, `load()`, `__str__()`

This structure ensures clarity and aligns code directly with visual pseudocode for better interpretability.

---

## 🚀 Installation

### 📦 Requirements

- Python 3.10 **(strictly required)**
- See `requirements.txt` for dependencies.

You can install dependencies using either pip or Anaconda.

#### Using pip

```bash
git clone https://github.com/yourusername/fund_rl.git
cd fund_rl
python3.10 -m venv venv
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

#### Using Anaconda

```bash
git clone https://github.com/yourusername/fund_rl.git
cd fund_rl
conda create -n fund_rl python=3.10
conda activate fund_rl
pip install -r requirements.txt
```