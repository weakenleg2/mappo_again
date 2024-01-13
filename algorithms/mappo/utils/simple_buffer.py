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

        self.buffer_size = args.episode_length
        self.actions_buffer = np.zeros((self.episode_length*self.num_agents*self.pretrain_dur, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.obs_buffer = np.zeros((self.episode_length*self.num_agents*self.pretrain_dur , self.n_rollout_threads, *obs_shape), dtype=np.float32)
        self.one_hot_list_buffer = np.zeros((self.episode_length*self.num_agents*self.pretrain_dur , self.n_rollout_threads, self.num_agents), dtype=np.float32)
        self.rewards_buffer = np.zeros((self.episode_length*self.num_agents*self.pretrain_dur , self.n_rollout_threads, 1), dtype=np.float32)
        self.current_size = 0
        self.step = 0

    def insert(self, action, obs, one_hot_list, reward):
        # print(type(one_hot_list))
        for agent_id in range(self.num_agents):
            # if self.current_size < self.buffer_size:
            self.actions_buffer[self.step] = action[:,agent_id].copy()
            self.obs_buffer[self.step] = np.array(list(obs[:, agent_id])).copy()
            self.one_hot_list_buffer[self.step] = one_hot_list[:,agent_id].detach().numpy().copy()
            self.rewards_buffer[self.step] = reward[:,agent_id].copy()
            self.current_size += 1
            self.step = (self.step + 1) % (self.episode_length*self.num_agents*self.pretrain_dur)
            # 倒是对的
            # print(self.step)
            # else:
            #     # Handle buffer overflow, for example, by removing the oldest data
            #     self.actions_buffer.pop(0)
            #     self.obs_buffer.pop(0)
            #     self.one_hot_list_buffer.pop(0)
            #     self.rewards_buffer.pop(0)

            #     self.actions_buffer.append(action)
            #     self.obs_buffer.append(obs)
            #     self.one_hot_list_buffer.append(one_hot_list)
            #     self.rewards_buffer.append(reward)

    def sample(self, batch_size):
        # Implement a method to sample a batch of data from the buffer
        pass

    def clear(self):
        self.actions_buffer.clear()
        self.obs_buffer.clear()
        self.one_hot_list_buffer.clear()
        self.rewards_buffer.clear()
        self.current_size = 0

# Usage
# replay_buffer = ReplayBuffer(buffer_size=10000)

# # Example of inserting data into the buffer
# for step in range(self.episode_length):
#     # ... (your existing code for generating actions, obs, one_hot_list, rewards)
#     replay_buffer.insert(actions, obs, one_hot_list, rewards)

# # Later, you can sample data from the replay buffer for training or analysis
