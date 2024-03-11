import torch
import numpy as np
from collections import defaultdict
from algorithms.mappo.utils.util import check, get_shape_from_obs_space, get_shape_from_act_space


class ReplayBuffer:
    def __init__(self, args,obs_space,  act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = args.num_agents
        self.pretrain_dur = args.pretrain_dur
        obs_shape = get_shape_from_obs_space(obs_space)
        # print("here",obs_shape)
        act_shape = get_shape_from_act_space(act_space)

        # self.buffer_size = args.episode_length
        self.obs_buffer = np.zeros((self.episode_length*self.pretrain_dur, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.next_obs_buffer = np.zeros_like(self.obs_buffer)
        self.actions_buffer = np.zeros((self.episode_length*self.pretrain_dur, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        self.one_hot_list_buffer = np.zeros((self.episode_length*self.pretrain_dur, self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
        self.rewards_buffer = np.zeros((self.episode_length*self.pretrain_dur, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.dones_buffer = np.zeros((self.episode_length*self.pretrain_dur, self.n_rollout_threads, self.num_agents, 1), dtype=np.bool_)
        self.current_size = 0
        self.step = 0

    def insert(self, action, obs, one_hot_list, reward,dones):
        # print(type(one_hot_list))
        # for agent_id in range(self.num_agents):
            # if self.current_size < self.buffer_size:
        #         self.obs_buffer[self.step] = np.array(list(obs)).copy()

        self.actions_buffer[self.step] = action.copy()
        self.obs_buffer[self.step] = np.array(list(obs)).copy()
        self.one_hot_list_buffer[self.step] = one_hot_list.detach().numpy().copy()
        self.rewards_buffer[self.step] = reward.copy()
        self.dones_buffer[self.step] = np.expand_dims(dones, -1).copy()
        # if self.step == 0:
        #     print(self.dones_buffer[self.step].all())
        
        if self.step > 0:
            for env_idx in range(self.n_rollout_threads):
                if not self.dones_buffer[self.step-1, env_idx].all():
                    # it means that the episode was still ongoing for at least one agent
                    self.next_obs_buffer[self.step-1, env_idx] = obs[env_idx].copy()
                    # print(self.next_obs_buffer[self.step-1, env_idx])
                else:
                    self.next_obs_buffer[self.step-1, env_idx] = np.zeros_like(obs[env_idx])
                    # print("done!")
        # print(self.dones_buffer[self.step].all())
        # print(self.next_obs_buffer[self.step])
        # Handle the last timestep of an episode
        # if self.step < (self.episode_length*self.pretrain_dur - 1) and dones.all():
        #     # If this is the last step of an episode, set the next obs to zero
        #     self.next_obs_buffer[self.step] = np.zeros_like(obs)

        # Update the step counter
        self.step = (self.step + 1) % (self.episode_length * self.pretrain_dur)

 