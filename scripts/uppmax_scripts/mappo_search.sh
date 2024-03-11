#!/bin/sh -l
#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p node 
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -J "mappo_search"
#SBATCH -o /home/pagliaro/project/RAC/outs/%j.out

PYTHON=/proj/uppmax2023-2-14/envs/RAC/bin/python3

module load conda
conda activate RAC
cd /home/pagliaro/project/RAC
$PYTHON scripts/mappo_search.py $@ --logdir ~/project/results 
