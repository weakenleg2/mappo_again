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


def _t2n(x):
    return x.detach().cpu().numpy()
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
            if episode == (self.pretrain_dur):
                cluster_idx = compute_clusters(self.easy_buffer, 
                                        self.all_args.num_agents,self.all_args.num_mini_batch*512, None, 
                                        1e-4, 10, 10, 0.0001, self.device)
                print(cluster_idx)
                sys.exit(0)
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
                # wandb.log({"com_savings":(tot_frames - tot_comms)/tot_frames},total_num_steps)

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

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.actor_hidden_size * 2), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads,
                             self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:,
                                                                                                agent_id],
                                                                                eval_masks[:,
                                                                                           agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(
                            self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate(
                                (eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.actor_hidden_size), dtype=np.float32)
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(
                np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append(
                {'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " %
                  agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents,
                                  self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                          rnn_states[:,
                                                                                     agent_id],
                                                                          masks[:,
                                                                                agent_id],
                                                                          deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(
                                self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate(
                                    (action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(
                            np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(
                    np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " %
                      agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif',
                            all_frames, duration=self.all_args.ifi)