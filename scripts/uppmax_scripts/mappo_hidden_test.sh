#!/bin/sh -l
#SBATCH -A uppmax2023-2-14
#SBATCH -M snowy
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -J "ra_mappo"
#SBATCH -o /home/pagliaro/project/RAC/outs/slurm.%j.out

PYTHON=/proj/uppmax2023-2-14/envs/RAC/bin/python3

env="MPE"
scenario="simple_spread" 
num_agents=$1
hidden_dim=$2
algo="mappo" #"mappo" 
seed=1
exp=hidden_size_comparision_norm_"$num_agents"

module load conda
conda activate RAC
cd /home/pagliaro/project/RAC
echo Running test with $1 agents and $2 nodes in hidden layer
$PYTHON scripts/train_mappo.py --env_name ${env} --algorithm_name ${algo} \
		--use_popart --use_valuenorm \
    --scenario_name ${scenario} --seed ${seed} --num_agents ${num_agents}\
    --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 40 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --hidden_dim ${size} --experiment_name ${exp} $@
