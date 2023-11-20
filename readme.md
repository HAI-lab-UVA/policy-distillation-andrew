# Policy Distillation
Implementation of supervised Actor-Critic policy distillation as a baseline for other transfer-learning RL approaches.

# Authors
* Andrew Balch - University of Virginia

# Getting Started (Ubuntu Linux)
1. Clone this repo
2. In a new virtual environment (conda recommended) run ```conda install --file requirements.txt``` or ```pip install -r requirements.txt```
3. To run pytorch GPU-accelerated, follow the instructions here https://pytorch.org/get-started/locally/
4. Clone Gymnasium to the source directory from https://github.com/Farama-Foundation/Gymnasium
5. Navigate to the directory where Gymnasium was cloned and run ```python setup.py install```