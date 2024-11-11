#!/bin/bash
#SBATCH -A m4776            
#SBATCH -C gpu               # Use GPU nodes
#SBATCH -q regular          # Use regular QOS
#SBATCH -t 00:30:00        # Request 30 minutes
#SBATCH -N 1               # Request 1 node
#SBATCH -n 1               # Request 1 task
#SBATCH --gpus 1           # Request 1 GPU
#SBATCH -c 32              # CPU cores per task
#SBATCH --gpu-bind=none    # Allow GPU binding

# Load required modules
module load nvidia/23.9
module load cuda/12.0.0

# Compile the GPU version
make gpu

# Run the program (example with some parameters)
srun ./build/gpu --nx 256 --ny 256 --scenario water_drop --num_iter 1000 --output output.bin