# ATC MARL Baselines

This repository is dedicated to the development and evaluation of Multi-Agent Reinforcement Learning (MARL) agents for the **Gym Air Traffic** environment.

## Project Structure

* `gym-air-traffic/`: The environment repository, included as a Git submodule.
* `agents/`: Implementation of various RL algorithms (PPO, MAPPO, etc.).
* `configs/`: Configuration files for training hyperparameters.
* `scripts/`: Utility scripts for training and visualization.

---

## Installation

### 1. Clone the Repository

Since the environment is a submodule, you must clone the repository with the `--recursive` flag to automatically fetch the environment code.

```bash
git clone --recursive https://github.com/TristanDonze/atc-marl-baselines.git
cd atc-marl-baselines

```

### 2. Setup the Environment

```bash
# Install the environment as a local editable package
pip install -e gym-air-traffic

# Install agent-specific dependencies (if any)
pip install -r requirements.txt

```

Using the `-e` (editable) flag ensures that any changes made to the code inside the `gym-air-traffic` folder are immediately reflected in your training scripts without needing a reinstall.
