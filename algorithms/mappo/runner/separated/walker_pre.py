import time
import wandb
import os
import numpy as np
import torch
from itertools import chain
from gymnasium.spaces.utils import flatdim

from algorithms.mappo.utils.util import update_linear_schedule
from algorithms.mappo.runner.separated.base_pre import Runner
from algorithms.mappo.algorithms.r_mappo.algorithm.ops_utils import compute_clusters

import imageio
import sys
import pandas as pd
from collections import Counter



def _t2n(x):
    return x.detach().cpu().numpy()
def normalize_indices(cluster_indices):
    normalized_indices = []
    for indices in cluster_indices:
        # Track the unique labels encountered in order and their normalized mappings
        unique_labels = {}
        next_label = 0
        normalized = []
        
        for index in indices:
            if index not in unique_labels:
                # Encounter a new label, assign it the next available normalized label
                unique_labels[index] = next_label
                next_label += 1
            
            # Map the original label to the normalized label
            normalized_label = unique_labels[index]
            normalized.append(normalized_label)
        
        normalized_indices.append(normalized)
    
    return normalized_indices

def find_most_common_index(index_counter):
    most_common = index_counter.most_common(2)  # Get the top 2 to check for a tie
    if most_common:
        # Check if there's more than one result and they have the same count
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            print("Tie detected. Choosing one arbitrarily.")
        # Extract the index from the list (choose the first one in case of a tie)
        most_common_index = list(most_common[0][0])
        return most_common_index
    else:
        return None

def count_normalized_indices(normalized_indices):
    # Convert the list of indices to a tuple so it can be used as a key in the counter
    index_tuples = [tuple(index) for index in normalized_indices]
    index_counter = Counter(index_tuples)
    return index_counter

def save_to_csv(data, num_agents, filename_prefix, episode, step):
    # act,obs,rew data
    num_instances, _, feature_size = data.shape
    for agent in range(num_agents):
        agent_data = data[:, agent, :].reshape(num_instances, -1)
        columns = [f'feature_{i+1}' for i in range(feature_size)]
        df = pd.DataFrame(agent_data, columns=columns)

        # Generate filename
        filename = f'{filename_prefix}_agent{agent+1}.csv'

        # Check if file exists to determine whether to write header
        file_exists = os.path.isfile(filename)

        # Append to CSV file (create if doesn't exist, append if it does)
        df.to_csv(filename, mode='a', index=False, header=not file_exists)

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def dict_to_tensor(self, x, iterable=True):
        #obs_shape = self.envs.observation_space('agent_0').shape
        if iterable:
          obs_shape = x[0]['agent_0'].shape
        else:
          obs_shape = ()

        output = np.zeros((len(x), self.num_agents, *obs_shape))
        for i, d in enumerate(x):
          d = list(d.values())
          d = np.array(d)
          output[i] = d

        return output

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads
        cluster_indices_list = []
        # print(episodes)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            tot_comms = 0
            tot_frames = 0
            # 要将学习的过程加入进去啊
            for step in range(self.episode_length):
                # print(self.episode_length)
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)
                # print(type(actions))
                # if step == 0:
                #     print(actions.shape)
                # # print(torch.nn.functional.one_hot(torch.tensor(actions,dtype=torch.int64), num_classes=actions.shape[-1]))
                # # Obser reward and next obs
                #     print(actions_env)
                one_hot_list = []
                for i in range(self.all_args.num_agents):
                    one_hot_agent = torch.nn.functional.one_hot(torch.tensor(i), self.all_args.num_agents)
                    one_hot_list.append(one_hot_agent)

                # Combine into a single tensor
                combined_matrix = torch.stack(one_hot_list)

                # Repeat the combined matrix to get a shape of (32, 3, 3)
                repeated_matrix = combined_matrix.repeat(self.all_args.n_rollout_threads, 1, 1)
                # if step ==0:
                #     print(repeated_matrix.shape)
                    

                # print(len(one_hot_list))
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # print(self.envs.get_cycle_count)
                # print("obs",obs)
                # print("rewards",rewards)
                # print("dones",dones)
                # print("infos",infos)

                obs = self.dict_to_tensor(obs)
                # if episode ==0:
                #     print("here",obs)
                # print(obs.shape)
                rewards = self.dict_to_tensor(rewards, False)
                # print(rewards.shape)
                rewards = np.expand_dims(rewards, -1)
                # dones = np.expand_dims(dones, -1)
                # print(type(actions),type(obs),type(rewards))
                # if 150 < episode <= (150+self.pretrain_dur):
                #     save_to_csv(np.clip(actions,-1,1), 3, 'actions', episode, step)
                #     save_to_csv(rewards, 3, 'rewards', episode, step)
                #     save_to_csv(obs, 3, 'observations', episode, step)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                
                for info in infos:
                    for agent_info in info.values():
                        tot_comms += agent_info['comms']
                        tot_frames += agent_info['frames']

                
                # insert data into buffer
                # if 150 < episode <= (150+self.pretrain_dur):
                    # 160
                # if step >=self.episode_length // 2:
                self.easy_buffer.insert(action=np.clip(actions,-1,1), obs=obs, one_hot_list=repeated_matrix, reward=rewards,dones=dones)
                # print()
                    # self.easy_buffer.insert(action=actions,obs=obs,one_hot_list=repeated_matrix,reward=rewards)
                self.insert(data)
            # print(self.easy_buffer.rewards_buffer)
            # compute return and update network
            # print(self.easy_buffer.one_hot_list_buffer)
            for iteration in range(5):
                if (episode == (self.pretrain_dur + iteration * 10)):

                    cluster_idx = compute_clusters(self.easy_buffer, 
                                                self.all_args.num_agents,
                                                self.vae_batch,
                                                self.clusters, 
                                                self.vae_lr, self.vae_epoch, self.vae_zfeatures, 
                                                self.kl, self.device)
                    # print(f"Iteration {iteration}, cluster_idx: {cluster_idx}")
                    cluster_indices_list.append(cluster_idx.cpu().numpy())

                if len(cluster_indices_list) == 5:
                    normalized_indices = normalize_indices(cluster_indices_list)
                    index_counter = count_normalized_indices(normalized_indices)
                    most_common_index = find_most_common_index(index_counter)
                    
                    
                    print("normalized_indices", normalized_indices)
                    print("most_common_index", most_common_index)
                    return most_common_index
                    
                    
                    # # Clear the list to start collecting new clustering results
                    # cluster_indices_list = []
                    # cluster_to_policy_index = {}
                    # for agent_id, cluster_idx in enumerate(most_common_index):
                    #     if cluster_idx not in cluster_to_policy_index:
                    #         # The first time we see this cluster index, we map it to the current agent's policy index
                    #         cluster_to_policy_index[cluster_idx] = agent_id

                    # # Step 2: Reassign policies based on cluster_to_policy_index
                    # for agent_id, cluster_idx in enumerate(most_common_index):
                    #     # Get the policy index of the first agent in this cluster
                    #     policy_index = cluster_to_policy_index[cluster_idx]
                    #     # Assign this policy to the current agent
                    #     self.policy[agent_id] = self.policy[policy_index]
                    # print(self.policy)
                    # print(f"Iteration {iteration}, most_common_index: {most_common_index}
                    # sys.exit(0)
            # for agent_id in range(self.num_agents):
            #     self.policy[agent_id].laac_sample= cluster_idx.repeat(self.n_rollout_threads, 1)
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * \
                self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\nScenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        #for info in infos:
                            #for count, info in enumerate(infos):
                                #if 'individual_reward' in infos[count][agent_id].keys():
                                    #idv_rews.append(infos[count][agent_id].get(
                                        #'individual_reward', 0))
                        #train_infos[agent_id].update(
                            #{'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(
                            self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)
                print('Average_episode_rewards: ', np.mean(self.buffer[0].rewards) * self.episode_length)
                # print((tot_frames - tot_comms)/tot_frames)
                wandb.log({"com_savings":(tot_frames - tot_comms)/tot_frames},total_num_steps)

            # eval
            # self.writter.add_scalar('communication_savings', 1 - tot_comms / (self.episode_length * self.num_agents * self.n_rollout_threads), episode)
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        obs = self.dict_to_tensor(obs)

        #last_actions = np.zeros(
          #(self.n_rollout_threads, self.num_agents * (flatdim(self.envs.action_space('agent_0')) - 1)))

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        #share_obs = np.concatenate([share_obs, last_actions], -1)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(
                list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            # print(self.buffer[agent_id].share_obs[step].shape)
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            action_space = self.envs.action_space('agent_' + str(agent_id))
            # print(action_space)
            if action_space.__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(
                        self.envs.action_space('agent_' + str(agent_id)).high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate(
                            (action_env, uc_action_env), axis=1)
            elif action_space.__class__.__name__ == 'Discrete':
                action_env = np.squeeze(
                    np.eye(action_space.n)[action], 1)
            else:
                action_env = np.clip(action, -1, 1)
            
            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = [{} for _ in range(self.n_rollout_threads)]
        for i in range(self.num_agents):
            thread_actions = temp_actions_env[i]
            for j in range(self.n_rollout_threads):
                actions_env[j]['agent_' + str(i)] = thread_actions[j]

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        # print(obs)
        # print("wuhu",np.array(list(obs)))
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.actor_hidden_size * 2), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.critic_hidden_size), dtype=np.float32)
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        #merged_actions = actions.reshape(self.n_rollout_threads, self.num_agents * (flatdim(self.envs.action_space('agent_0')) - 1))
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        #share_obs = np.concatenate([share_obs, merged_actions], -1)
        # print(rewards.shape)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])
            # print(self.buffer[0].rewards.shape)

    