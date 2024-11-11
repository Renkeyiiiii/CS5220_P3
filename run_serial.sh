#!/bin/bash
#SBATCH -A m4776            
#SBATCH --qos=shared
#SBATCH -t 00:10:00        # Request 10 minutes
#SBATCH -N 1               # Request 1 node
#SBATCH -n 1               # Request 1 task
#SBATCH -c 32              # CPU cores per task


# Compile the GPU version
make gpu

# Run the program (example with some parameters)
srun ./build/gpu --nx 256 --ny 256 --scenario water_drop --num_iter 1000 --output serial.out