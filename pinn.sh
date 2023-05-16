#!/bin/bash

# Job Name
#SBATCH -J PINN

# Time Requested (Not yet tested)
#SBATCH -t 1:00:00

# Number of Cores (max 4)
#SBATCH -c 2

# Single Node (Not yet optimized for Distributed Computing)
#SBATCH -N 1

# Request GPU partition and Access (max 2)
#SBATCH -p gpu --gres=gpu:1

# Request Memory (Not yet tested)
#SBATCH --mem=10G

# Outputs
#SBATCH -e ./scratch/PINN.err
#SBATCH -o ./scratch/PINN.out

########### END OF SLURM COMMANDS ##############

# Show CPU infos
lscpu

# Show GPU infos
nvidia-smi

# Force Printing Out `stdout`
export PYTHONBUFFERED=TRUE

# Load Modules for JAX venv
module load python/3.11.0 openssl/3.0.0 cuda/11.7.1 cudnn/8.6.0

# Activate JAX venv
source ../../jax.venv/bin/activate

# Run the Python file with arguments
python3 train_PINN.py --inverseprob True --savefig True --savemodel True --lamda 10.0,10.0,1.0 --lbfgs 1 --adam 8000

