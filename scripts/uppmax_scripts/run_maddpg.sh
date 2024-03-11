#!/bin/bash -l

#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J "ra_maddpg"
#SBATCH -o /home/pagliaro/project/private/RAC/outs/slurm.%j.out

PYTHON=/proj/uppmax2023-2-14/envs/RAC/bin/python3

module load conda
conda activate RAC
cd /home/pagliaro/project/private/RAC

$PYTHON run_ra_maddpg.py $@
