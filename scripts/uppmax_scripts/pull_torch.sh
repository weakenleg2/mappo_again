#!/bin/bash -l

#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -J "Torch install"

singularity pull docker://nvcr.io/nvidia/pytorch:22.03-py3
