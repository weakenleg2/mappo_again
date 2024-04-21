import time
import wandb
import os
import numpy as np
import torch
from itertools import chain
from gymnasium.spaces.utils import flatdim

from algorithms.mappo.utils.util import update_linear_schedule
from algorithms.mappo.runner.separated.base_dpo_simp import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()
class MinMaxPenalty():
    """
    Learn the highest reward/penalty that minimises the probability of reaching bad terminal states
    # here the dangerous state is communication requirement exceeds bandwidth
    Init:
        - rmin (optional): The lower bound for environment rewards
        - rmax (optional): The upper bound for environment rewards   
    Update:
        - Returns the minmax penalty estimate
    Usage:
    Symlink to the desired folder and import, or copy-paste to where needed in the code
    In training loop: 
        minmaxpenalty = MinMaxPenalty()
        for each step:
            - take an action and get reward and q_value (or just [value] if RL algorithm only learns state_values)
            penalty = minmaxpenalty.update(reward, Q[state])
            if info["unsafe"]:
                reward = penalty
    """

    # def __init__(self, rmin=0, rmax=0):
    #     self.rmin = rmin
    #     self.rmax = rmax
    #     self.vmin = self.rmin
    #     self.vmax = self.rmax
    #     self.penalty = min([self.rmin, (self.vmin-self.vmax)])
    
    # def update(self, reward, value):
    #     self.rmin = min([self.rmin, reward])
    #     self.rmax = max([self.rmax, reward])
    #     self.vmin = min([self.vmin, self.rmin, min(value)])
    #     self.vmax = max([self.vmax, self.rmax, max(value)])

    #     self.penalty = min([self.rmin, (self.vmin-self.vmax)])

    #     return self.penalty
    # def __init__(self, shape=(14, 3, 1)):
    #     # Initialize min/max trackers to infinity with the shape of the input arrays
    #     self.rmin = np.full(shape, 0)
    #     self.rmax = np.full(shape, 0)
    #     self.vmin = self.rmin
    #     self.vmax = self.rmax
    #     # self.penalty = min([self.rmin, (self.vmin-self.vmax)])
    
    # def update(self, reward, value):
    #     # Ensure input is numpy arrays for compatibility
    #     reward = np.array(reward)
    #     value = np.array(value)
        
    #     # Element-wise update of rmin, rmax, vmin, vmax
    #     self.rmin = np.minimum(self.rmin, reward)
    #     self.rmax = np.maximum(self.rmax, reward)
    #     self.vmin = np.minimum(self.vmin, value)
    #     self.vmax = np.maximum(self.vmax, value)
        
    #     # Calculate penalties element-wise
    #     self.penalty = np.minimum(self.rmin, self.vmin - self.vmax)
        
    #     return self.penalty

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
        rewardupdate = MinMaxPenalty()
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            tot_comms = 0
            tot_frames = 0
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs,log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)

                # Obser reward and next obs
                # print(actions_env)

                obs, rewards, dones, infos = self.envs.step(actions_env)
                

                obs = self.dict_to_tensor(obs)
                # print(obs.shape)
                rewards = self.dict_to_tensor(rewards, False)
                rewards = np.expand_dims(rewards, -1)
                # print(rewards.shape,values.shape)

                # rewards = rewardupdate.update(rewards, values)


                data = obs, rewards, dones, infos, values, actions, action_log_probs,log_probs, rnn_states, rnn_states_critic
                
                for info in infos:
                    for agent_info in info.values():
                        tot_comms += agent_info['comms']
                        tot_frames += agent_info['frames']


                # insert data into buffer
                self.insert(data)

            # compute return and update network
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
        # print("obs",obs)
        # print("state",self.envs.state())
        # print(obs.shape)

        #last_actions = np.zeros(
          #(self.n_rollout_threads, self.num_agents * (flatdim(self.envs.action_space('agent_0')) - 1)))
        # if not self.use_centralized_V:
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
        log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            log_prob = self.trainer[agent_id].policy.get_probs(self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
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
            # print(action.shape,action_log_prob.shape)
            action_log_probs.append(_t2n(action_log_prob))
            log_probs.append(_t2n(log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        # print(temp_actions_env)
        actions_env = [{} for _ in range(self.n_rollout_threads)]
        for i in range(self.num_agents):
            thread_actions = temp_actions_env[i]
            for j in range(self.n_rollout_threads):
                actions_env[j]['agent_' + str(i)] = thread_actions[j]

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        log_probs = np.array(log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs,log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs,log_probs, rnn_states, rnn_states_critic = data
        # print("log_probs",log_probs)
        # rnn_states[dones == True] = np.zeros(
        #     ((dones == True).sum(), self.recurrent_N, self.actor_hidden_size * 2), dtype=np.float32)
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.actor_hidden_size), dtype=np.float32)
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

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])