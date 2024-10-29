# Job Shop Scheduling with DQN and APO Optimization

This repository implements a Deep Q-Network (DQN) for job shop scheduling and an Adaptive Protozoa Optimization (APO) for benchmark optimization tasks, incorporating CEC2022 functions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [DQN Job Shop Scheduling](#dqn-job-shop-scheduling)
  - [APO Benchmark Optimization](#apo-benchmark-optimization)
- [Results](#results)
- [File Structure](#file-structure)

## Overview

This project addresses:
1. **Job Shop Scheduling**: Using DQN and transformers to optimize dynamic job allocation to machines.
2. **Benchmark Function Optimization**: Applying the APO algorithm for optimization on CEC2022 benchmark functions.

## Features

### 1. DQN for Job Shop Scheduling
- **Attention Mechanisms**: Custom multi-head attention layers to focus on critical job features.
- **Transformer Encoder**: Encodes scheduling inputs to handle complex interdependencies.
- **Flexible Scheduling Rules**: Supports nine scheduling rules to optimize job assignment.

### 2. APO for Benchmark Optimization
- **CEC2022 Benchmark Compatibility**: Optimizes 12 benchmark functions.
- **Adaptive Learning Rate and Epsilon**: Protozoa-based population adapts dynamically for convergence.

## Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/yourusername/job-shop-scheduling-dqn-apo.git
cd job-shop-scheduling-dqn-apo
pip install -r requirements.txt
```

## Usage

### DQN Job Shop Scheduling

The `DQN.py` file contains the job shop scheduling model. Configure the job shop environment and start training:

```python
from DQN import train

# Example configuration
Total_Machine = [8, 12, 16]
Job_insert = [20, 30, 40]
E_ave = [50, 100, 200]
machine = Total_Machine[0]
e_ave = E_ave[0]
job_insert = Job_insert[2]

train(machine, e_ave, job_insert)
```

Results (e.g., total tardiness, machine utilization, makespan) are saved as Excel files under the `results` directory.

### APO Benchmark Optimization

The `APO_func.py` file contains the APO algorithm for benchmark optimization. Run APO with customized parameters:

```python
from APO_func import APO_func

# APO function configuration
Fid = 1
dim = 2
pop_size = 100
iter_max = 200
Xmin = -100
Xmax = 100

# Run optimization
bestFit, elapsedTime, best_solution, best_learning_rate, best_epsilon, fitness_history = APO_func(
    Fid, dim, pop_size, iter_max, Xmin, Xmax
)
```

The results, including fitness score, best solution, learning rate, and epsilon, are saved to an Excel file.

## Results

Results for DQN scheduling and APO optimization are saved to Excel files in the `results` folder. Use the saved data to analyze performance metrics like total reward, makespan, and fitness history.

## File Structure

```plaintext
├── DQN.py               # DQN model for job shop scheduling
├── APO_func.py          # APO optimization function
├── Job_shop.py          # Job shop environment and scheduling rules
├── cec2022.py           # CEC2022 benchmark functions
├── requirements.txt     # List of required Python packages
├── results/             # Directory for saving results in Excel files
└── README.md            # Project overview and usage instructions
```

---

This `README.md` should provide a clear and organized overview of your repository's functionality, usage, and setup instructions. Adjust file paths and project links as needed!
