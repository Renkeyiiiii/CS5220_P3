#!/bin/bash
#SBATCH -J serial
#SBATCH -o serial_%j.out
#SBATCH -e serial_%j.err
#SBATCH -A m4776
#SBATCH -C cpu
#SBATCH -c 32
#SBATCH --qos=debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1


# Compile the CPU version
make basic_serial

# Run the program (example with some parameters)
srun ./build/basic_serial --nx 256 --ny 256 --scenario water_drop --num_iter 1000 --output serial.out
