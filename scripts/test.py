import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from custom_envs.multiwalker_communicate import multiwalker_com
from algorithms.mappo.config import get_config
# from algorithms.mappo.envs.mpe.MPE_env import MPEEnv
from algorithms.mappo.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv
env = multiwalker_com.parallel_env(n_walkers=3, position_noise=0, 
                                                angle_noise=0, forward_reward=8.0, terminate_reward=-100.0,
                                                fall_reward=-10.0, shared_reward=False,
                                                terminate_on_fall=True,remove_on_fall=True,
                                                terrain_length=200,
                                                penalty_ratio=0.1,
                                                full_comm=True,
                                                delay = 0,
                                                packet_drop_prob = 0,
                                                max_cycles=500)
env.reset()
print(env.state())