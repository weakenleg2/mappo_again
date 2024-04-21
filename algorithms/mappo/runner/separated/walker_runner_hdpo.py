import time
import wandb
import os
import numpy as np
import torch
from itertools import chain
from gymnasium.spaces.utils import flatdim

from algorithms.mappo.utils.util import update_linear_schedule
from algorithms.mappo.runner.separated.base_hdpo import Runner
from algorithms.mappo.algorithms.r_mappo.algorithm.ops_utils import compute_clusters
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



class PolicyBlender:
    def __init__(self, policies, cluster_indices):
        self.policies = policies  # A list of all policy networks
        self.cluster_indices = cluster_indices  # List of cluster indices for each policy
        self.alpha = 0.0
        self.blend_threshold = 0.9
        self.increment = 0.05

    def blend_policies(self):
        if self.alpha < self.blend_threshold:
            for cluster_idx in set(self.cluster_indices):
                target_policy_idx = self.cluster_indices.index(cluster_idx)
                target_policy = self.policies[target_policy_idx]
                for agent_id, idx in enumerate(self.cluster_indices):
                    if idx == cluster_idx and agent_id != target_policy_idx:
                        policy = self.policies[agent_id]
                        for target_param, param in zip(target_policy.actor.parameters(), policy.actor.parameters()):
                            param.data.copy_((1 - self.alpha) * param.data + self.alpha * target_param.data)
                        for target_param, param in zip(target_policy.critic.parameters(), policy.critic.parameters()):
                            param.data.copy_((1 - self.alpha) * param.data + self.alpha * target_param.data)

    def update_alpha(self):
        self.alpha += self.increment
        


class MPERunner(Runner):
    def __init__(self, most_common_index,config):
        super(MPERunner, self).__init__(most_common_index,config)

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
        most_common_index2 = None

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            tot_comms = 0
            tot_frames = 0
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, penalty_values, rnn_states_penalty = self.collect(
                    step)
                # print("penalty_values",penalty_values.shape)
                # Obser reward and next obs
                # print(actions_env)
                one_hot_list = []
                for i in range(self.all_args.num_agents):
                    one_hot_agent = torch.nn.functional.one_hot(torch.tensor(i), self.all_args.num_agents)
                    one_hot_list.append(one_hot_agent)

                # Combine into a single tensor
                combined_matrix = torch.stack(one_hot_list)

                # Repeat the combined matrix to get a shape of (32, 3, 3)
                repeated_matrix = combined_matrix.repeat(self.all_args.n_rollout_threads, 1, 1)

                obs, rewards, dones, infos = self.envs.step(actions_env)
                

                obs = self.dict_to_tensor(obs)
                # print(obs.shape)
                rewards = self.dict_to_tensor(rewards, False)
                rewards = np.expand_dims(rewards, -1)
                # print(rewards.shape,values.shape)
                communication_actions = actions[:, :, -1]
                communication_mask = communication_actions == 1
                communication_penalty = np.expand_dims(communication_mask, -1) * penalty_values
                rewards = rewards - communication_penalty
                # print("communication_penalty",communication_penalty)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic,penalty_values, rnn_states_penalty,  communication_penalty
                
                for info in infos:
                    for agent_info in info.values():
                        tot_comms += agent_info['comms']
                        tot_frames += agent_info['frames']

                self.easy_buffer.insert(action=np.clip(actions,-1,1), obs=obs, one_hot_list=repeated_matrix, reward=rewards,dones=dones)
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            for iteration in range(5):

                if (episode == (self.pretrain_dur+self.mid_gap + iteration * 10)):

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
                        most_common_index2 = find_most_common_index(index_counter)
                        
                        
                        print("normalized_indices", normalized_indices)
                        print("most_common_index", most_common_index2)
                        policy_blenders = PolicyBlender(self.policy, most_common_index2)

                        
            if most_common_index2 is not None:
                
                if policy_blenders.alpha < policy_blenders.blend_threshold:
                    # print("blend here")
                    policy_blenders.blend_policies()
                    policy_blenders.update_alpha()
                            
                if policy_blenders.alpha >= (policy_blenders.blend_threshold-0.1):
                    # print("blending done!")
                    cluster_to_policy_index = {}
                    for agent_id, cluster_idx in enumerate(most_common_index2):
                        if cluster_idx not in cluster_to_policy_index:
                            # The first time we see this cluster index, we map it to the current agent's policy index
                            cluster_to_policy_index[cluster_idx] = agent_id
                    for agent_id, cluster_idx in enumerate(most_common_index2):
                        # Get the policy index of the first agent in this cluster
                        policy_index = cluster_to_policy_index[cluster_idx]
                        # Assign this policy to the current agent
                        self.policy[agent_id] = self.policy[policy_index]
                        cluster_indices_list = []
                        most_common_index2 = None
                
                # return most_common_index
                
                
                # Clear the list to start collecting new clustering results
                
                    
                    print(self.policy)
            self.compute()
            train_infos = self.train()
            # self.render()

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
                # print(np.mean(self.buffer[0].rewards))
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
        # print(obs.shape)
        obs = self.dict_to_tensor(obs)
        
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        #share_obs = np.concatenate([share_obs, last_actions], -1)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            elif self.use_centralized_V:
                share_obs = self.envs.state()
                # Slice up to the end of the current agent's observation
                # share_obs_1 = share_obs[:, :(agent_id+1)*23]
                # # print("obs1",share_obs_1.shape)
                # # Slice from the start of the next agent's observation
                # share_obs_2 = share_obs[:, (agent_id+1)*23:]
                # print("obs2",share_obs_2.shape)
                agent_obs = np.array(list(obs[:, agent_id]))
                # print("last6",agent_obs_last_6.shape)
                # Concatenate: first part + last 6 elements + second part
                share_obs = np.concatenate((share_obs, agent_obs), axis=-1)

                # print("share_obs",share_obs.shape)
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(
                list(obs[:, agent_id])).copy()
        # else self.use_centralized_V:


    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        # log_probs = []
        rnn_states = []
        rnn_states_critic = []
        penalty_values = []
        rnn_states_penalty = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # log_prob = self.trainer[agent_id].policy.get_probs(self.buffer[agent_id].obs[step],
            #                                                 self.buffer[agent_id].rnn_states[step],
            #                                                 self.buffer[agent_id].masks[step])
            penalty_value,rnn_state_penalty = self.trainer[agent_id].policy.get_penalty(self.buffer[agent_id].share_obs[step],
                                                                      self.buffer[agent_id].rnn_states_penalty[step],
                                                                    self.buffer[agent_id].masks[step]
                                                                      )
            # [agents, envs, dim]
            values.append(_t2n(value))
            penalty_values.append(_t2n(penalty_value))
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
            # print(action.shape,action_log_prob.shape)
            action_log_probs.append(_t2n(action_log_prob))
            # log_probs.append(_t2n(log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))
            rnn_states_penalty.append(_t2n(rnn_state_penalty))

        # [envs, agents, dim]
        # print(temp_actions_env)
        actions_env = [{} for _ in range(self.n_rollout_threads)]
        for i in range(self.num_agents):
            thread_actions = temp_actions_env[i]
            for j in range(self.n_rollout_threads):
                actions_env[j]['agent_' + str(i)] = thread_actions[j]

        values = np.array(values).transpose(1, 0, 2)
        penalty_values = np.array(penalty_values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        # log_probs = np.array(log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)
        rnn_states_penalty = np.array(rnn_states_penalty).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env,\
                penalty_values, rnn_states_penalty

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, penalty_values, rnn_states_penalty,communication_penalty = data
        # print("log_probs",log_probs)
        # rnn_states[dones == True] = np.zeros(
        #     ((dones == True).sum(), self.recurrent_N, self.actor_hidden_size * 2), dtype=np.float32)
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.actor_hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.critic_hidden_size), dtype=np.float32)
        rnn_states_penalty[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.critic_hidden_size), dtype=np.float32)
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)
        # penalty_rewards = 
        # penalty_reward = np.zeros_like(rewards, dtype=float)
        #merged_actions = actions.reshape(self.n_rollout_threads, self.num_agents * (flatdim(self.envs.action_space('agent_0')) - 1))
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        #share_obs = np.concatenate([share_obs, merged_actions], -1)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            elif self.use_centralized_V:
                share_obs = self.envs.state()

                # share_obs_1 = share_obs[:, :(agent_id+1)*23]
                # # Slice from the start of the next agent's observation
                # share_obs_2 = share_obs[:, (agent_id+1)*23:]
                
                # Extract the last 6 elements from the current agent's observation and reshape if necessary
                agent_obs = np.array(list(obs[:, agent_id]))
                
                # Concatenate: first part + last 6 elements + second part
                share_obs = np.concatenate((share_obs, agent_obs), axis=-1)
            # penalty_reward[actions[:, :, -1] == 1] = -self.all_args.com_ratio

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id],
                                         communication_penalty[:,agent_id],
                                         rnn_states_penalty[:,agent_id],
                                         penalty_values[:, agent_id]
                                         )