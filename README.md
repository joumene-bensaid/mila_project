# Direction-Aware Fusion

This project implements various model fusion strategies for continual learning tasks, with a focus on direction-aware approaches to mitigate catastrophic forgetting.

## Overview

The project explores different fusion methods for combining fine-tuned models:
- **Soft Soup**: Simple weighted averaging of model parameters
- **Orthogonal Deltas**: Direction-aware fusion using orthogonal decomposition
- **DoRA (Directional Representation Alignment)**: Advanced direction-aware fusion

## Project Structure

```
direction_aware_fusion/
├── src/                    # Source code
│   ├── fusion/            # Fusion method implementations
│   ├── modeling/          # Training and evaluation utilities
│   └── utils/             # Configuration and utility functions
├── experiments/           # Experiment scripts
├── scripts/              # Shell scripts for running experiments
├── assets/               # Organized outputs and artifacts
│   ├── logs/            # Experiment logs
│   ├── wandb/           # Weights & Biases tracking data
│   └── plots/           # Generated plots and visualizations
└── requirements.txt      # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables for Weights & Biases (optional):
```bash
export WANDB_API_KEY=your_wandb_api_key
```

## Usage

### Running Experiments

The project includes several experiment scripts:

1. **Test 0 - Soft Soup Baseline**:
```bash
python experiments/test0_soft_soup.py
```

2. **Test 1 - Orthogonal Deltas**:
```bash
python experiments/test1_orthogonal_deltas.py
```

3. **Test 2 - DoRA**:
```bash
python experiments/test2_dora.py
```

### Running All Tests

Use the provided shell scripts:
```bash
# Run all experiments
./scripts/run_all.sh

# Run individual tests
./scripts/run_test0.sh
./scripts/run_test1.sh
```

### SLURM Integration

For cluster environments, use the job submission script:
```bash
sbatch experiments/job_submit.sh
```

## Fusion Methods

### 1. Soft Soup (`src/fusion/soft.py`)
Simple parameter averaging with configurable weights.

### 2. Orthogonal Deltas (`src/fusion/orthogonal.py`)
Direction-aware fusion that decomposes parameter deltas into orthogonal components.

### 3. DoRA (`src/fusion/dora.py`)
Advanced directional representation alignment for improved fusion quality.

## Configuration

Experiments are configured through `src/utils/config.py`. Key parameters include:
- Model architectures and datasets
- Training hyperparameters
- Fusion method settings
- Logging and tracking options

## Outputs

All experiment outputs are organized in the `assets/` folder:

- **Logs**: Training logs, error logs, and experiment outputs
- **Wandb**: Weights & Biases tracking data and artifacts
- **Plots**: Generated visualizations and analysis plots

## Results

The project evaluates continual learning performance using:
- Task-specific accuracy metrics
- Catastrophic forgetting measures
- Cross-task interference analysis

## Dependencies

- PyTorch >= 1.12.0
- Transformers >= 4.20.0
- Datasets >= 2.0.0
- Scikit-learn >= 1.0.0
- NumPy >= 1.21.0
- Weights & Biases >= 0.12.0

## Contributing

1. Follow the existing code structure
2. Add appropriate tests for new fusion methods
3. Update documentation for new features
4. Ensure experiments are reproducible

## License

[Add your license information here]

## Citation

[Add citation information if applicable]
