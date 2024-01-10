#!/usr/bin/env python
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
from algorithms.mappo.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
# import gym
from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from torch.distributions import Categorical
from collections import defaultdict
from collections import deque
from algorithms.mappo.algorithms.r_mappo.algorithm.ops_utils import compute_clusters
from algorithms.mappo.algorithms.r_mappo.algorithm.model import Policy
from gymnasium.spaces.utils import flatdim
# from gym.spaces import Box, Tuple


def _compute_returns(storage, next_value, gamma):
    returns = [next_value]
    for rew, done in zip(reversed(storage["rewards"]), reversed(storage["done"])):
        ret = returns[0] * gamma + rew * (1 - done.unsqueeze(1))
        returns.insert(0, ret)

    return returns
def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info
def _compute_loss(model, storage, value_loss_coef, entropy_coef, central_v):
    with torch.no_grad():
        next_value = model.get_value(storage["state" if central_v else "obs"][-1])
    returns = _compute_returns(storage, next_value)

    input_obs = zip(*storage["obs"])
    input_obs = [torch.stack(o)[:-1] for o in input_obs]

    if central_v:
        input_state = zip(*storage["state"])
        input_state = [torch.stack(s)[:-1] for s in input_state]
    else:
        input_state = None

    input_action_mask = zip(*storage["action_mask"])
    input_action_mask = [torch.stack(a)[:-1] for a in input_action_mask]

    input_actions = zip(*storage["actions"])
    input_actions = [torch.stack(a) for a in input_actions]

    values, action_log_probs, entropy = model.evaluate_actions(
        input_obs, input_actions, input_action_mask, input_state,
    )

    returns = torch.stack(returns)[:-1]
    advantage = returns - values

    actor_loss = (
        -(action_log_probs * advantage.detach()).sum(dim=2).mean()
        - entropy_coef * entropy
    )
    value_loss = (returns - values).pow(2).sum(dim=2).mean()

    loss = actor_loss + value_loss_coef * value_loss
    return loss
n_rollout_threads = 32
def make_train_env():
    def get_env_fn(rank):
        def init_env():
            env = multiwalker_com.parallel_env(n_walkers=3, position_noise=0, 
                                                angle_noise=0, forward_reward=5.0, 
                                                terminate_reward=-100.0,
                                                fall_reward=-10.0, shared_reward=False,
                                                terminate_on_fall=True,remove_on_fall=True,
                                                terrain_length=200,
                                                penalty_ratio=0.05,
                                                full_comm=False,max_cycles=500)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(32)])
envs = make_train_env()
# print(envs.action_space('agent_0' ))
agent_count = 3
obs_size = envs.observation_space('agent_0').shape
action_space = envs.action_space('agent_0')
# print(action_space)
act_size = 0
for space in action_space:
    # print(space)
    if space.__class__.__name__ == 'Box':
        # Add the dimensions of the Box space
        # print("true")
        act_size += space.shape[0]
    elif space.__class__.__name__ == 'Discrete':
        # Treat the Discrete space as having a single dimension
        act_size += 1
# print(agent_count,obs_size,act_size)
env_dict = {
    "obs": {"shape": obs_size, "dtype": np.float32},
    "rew": {"shape": 1, "dtype": np.float32},
    "next_obs": {"shape": obs_size, "dtype": np.float32},
    "done": {"shape": 1, "dtype": np.float32},
    "act": {"shape": act_size, "dtype": np.float32},
    "agent": {"shape": agent_count, "dtype": np.float32},
}
pretraining_steps = 5000
n_steps = 5 #not env steps
rb = ReplayBuffer(int(agent_count * 5000 * n_rollout_threads * 5), env_dict)
algorithm_mode = "ops"
clusters = None
if clusters:
    model_count = clusters
else:
    model_count = min(10, agent_count)
laac_size = model_count
# print(laac_size)
# laac_params = torch.nn.Parameter(torch.ones(3-1, laac_size))
# print(self)
batch_size = 1
# sample = Categorical(logits=laac_params).sample([batch_size])
# laac_sample = torch.cat((torch.zeros(batch_size,1).int(), sample), dim=1)
# print("laac_sample1",laac_sample)
obs = envs.reset()
device = "cpu"
architecture = {
        "actor": [64, 64],
        "critic": [64, 64],
    }
state_size = None
from gymnasium import spaces
combinedobs_space = spaces.Tuple([envs.observation_space('agent_0'),envs.observation_space('agent_0')
                               ,envs.observation_space('agent_0')
                ])

combinedac_space = spaces.Tuple([envs.action_space('agent_0'),envs.action_space('agent_0')
                               ,envs.action_space('agent_0')
                ])
model = Policy(combinedobs_space, combinedac_space, architecture, model_count, state_size)
model.to(device)
lr = 3e-4
optim_eps = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr, eps=optim_eps)
storage = defaultdict(lambda: deque(maxlen=n_steps))
storage["obs"] = deque(maxlen=n_steps + 1)
storage["done"] = deque(maxlen=n_steps + 1)
storage["obs"].append(obs)
storage["done"].append(torch.zeros(n_rollout_threads))
storage["info"] = deque(maxlen=10)
model.sample_laac(n_rollout_threads)
# print(model.laac_sample)
if algorithm_mode == "ops":
    model.laac_sample = torch.zeros(n_rollout_threads, agent_count).long()
# print("laac_sample2",laac_sample)
total_steps = int(10e6)
delay = 0 
pretraining_times = 1
delay_training = False
# use_proper_termination = true
def dict_to_tensor(x, iterable=True):
    #obs_shape = self.envs.observation_space('agent_0').shape
    x = x[0]
    if iterable:
        obs_shape = x[0]['agent_0'].shape
    else:
        obs_shape = ()

    output = np.zeros((len(x),agent_count , *obs_shape))
    for i, d in enumerate(x):
        # print(d)
        d = list(d.values())
        d = np.array(d)
        output[i] = d

    return torch.from_numpy(output)
for step in range(total_steps):
        
    if algorithm_mode == "ops" and step in [delay + pretraining_steps*(i+1) for i in range(pretraining_times)]:
        print(f"Pretraining at step: {step}")
        cluster_idx = compute_clusters(rb.get_all_transitions(), agent_count,128,clusters=clusters,
                                       lr=3e-4,epochs=10,z_features=10,kl_weight=0.0001)
        model.laac_sample = cluster_idx.repeat(n_rollout_threads, 1)
        # print(model.laac_sample)
        # pickle.dump(rb.get_all_transitions(), open(f"{env_name}.p", "wb"))
        # _log.info(model.laac_sample)
    for n_step in range(n_steps):
        with torch.no_grad():
            # tensor_storage = deque()

            # for numpy_array in storage["obs"]:
            #     array_of_tensors = []
            #     for dictionary in numpy_array:
            #         dict_tensors = {key: torch.from_numpy(value) for key, value in dictionary.items()}
            #         array_of_tensors.append(dict_tensors)
            #     tensor_storage.append(array_of_tensors)
            # print(tensor_storage)
            # tensor_storage = {agent: torch.from_numpy(storage["obs"][-1]) for agent, obs in storage.items()}
            # print([torch.tensor(obs) for obs in storage["obs"][-1]])
            storage["obs"] = dict_to_tensor(storage["obs"])
            # print(storage["obs"][-1])
            # print(storage["obs"][-1])
            actions = model.act(storage["obs"])
        (obs, state, action_mask), reward, done, info = envs.step(actions)

        # if use_proper_termination:
        #     bad_done = torch.FloatTensor(
        #         [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
        #     ).to(device)
        #     done = done - bad_done

        storage["obs"].append(obs)
        storage["actions"].append(actions)
        storage["rewards"].append(reward)
        storage["done"].append(done)
        storage["info"].extend([i for i in info if "episode_reward" in i])
        storage["laac_rewards"] += reward

        if algorithm_mode == "ops" and step < delay + pretraining_times * pretraining_steps:
            for agent in range(len(obs)):

                one_hot_action = torch.nn.functional.one_hot(actions[agent], act_size).squeeze().numpy()
                one_hot_agent = torch.nn.functional.one_hot(torch.tensor(agent), agent_count).repeat(parallel_envs, 1).numpy()

                # if bad_done[0]:
                #     nobs = info[0]["terminal_observation"]
                #     nobs = [torch.tensor(o).unsqueeze(0) for o in nobs]
                # else:
                nobs = obs
                    
                data = {
                    "obs": storage["obs"][-2][agent].numpy(),
                    "act": one_hot_action,
                    "next_obs": nobs[agent].numpy(),
                    "rew":  reward[:, agent].unsqueeze(-1).numpy(),
                    "done": done[:].unsqueeze(-1).numpy(),
                    # "policy": np.array([model.laac_sample[0, agent].float().item()]),
                    "agent": one_hot_agent,
                    # "timestep": step,
                    # "nstep": n_step,
                }
                rb.add(**data)
                
    if algorithm_mode == "ops" and step < pretraining_steps and delay_training:
        continue
    # if laac_mode=="laac" and step and step % laac_timestep == 0:
    #     loss += _compute_laac_loss(model, storage)
    loss = _compute_loss(model, storage)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    # df["agent"] = df["agent"].astype(int)
    # df["timestep"] = df["timestep"].astype(int)
    # df = df.set_index(["timestep", "agent"])

envs.close()