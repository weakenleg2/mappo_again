import time
import wandb
import numpy as np
from functools import reduce
import torch
from algorithms.mappo.runner.shared.base_runner_dpo import Runner
import wandb
import imageio
from gymnasium.spaces.utils import flatdim
import ray

from ray.air import session

def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
    

    def dict_to_tensor(self, x, iterable = True):
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
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            if self.args.multi_rollout:
                for rollout in range(self.all_args.n_rollout_threads):
                    self.warmup(rollout)
                    for step in range(self.episode_length):
                        # Sample actions
                        base_ret, q_ret, sp_ret,penalty_ret = self.collect(step, rollout)
                        values, actions, action_log_probs, rnn_states, rnn_states_critic,action_env = base_ret
                        # Obser reward and next obs
                        # print(action_env.shape)
                        acts = {}
                        actions_env = []
                        for j in range(self.num_agents):
                            acts['agent_' + str(j)] = np.squeeze(action_env[:, j, :])
                        actions_env.append(acts)
                        # print("actions_env",actions_env)
                        obs, rewards, dones, infos = self.envs.step(actions_env)
                        # print(obs.shape)
                        obs = self.dict_to_tensor(obs)
                        rewards = self.dict_to_tensor(rewards, False)
                        rewards = np.expand_dims(rewards, -1)
                        available_actions = None
                        # if self.args.mpe_share_obs:
                        #     tmp_shape = np.shape(obs)
                        #     share_obs = obs.reshape([tmp_shape[0],1,-1 ]).repeat(self.args.n_agents,axis = 1)
                        # else:
                        share_obs = obs
                        
                        # print('rs = {}'.format(self.args.reward_scale_const))
                        
                            # print('after scale: reward = {}'.format(rewards))

                        base_input = [obs, share_obs, rewards, dones, infos, available_actions, \
                                      values, actions, action_log_probs, \
                                      rnn_states, rnn_states_critic]
                        data = [base_input, q_ret, sp_ret,penalty_ret]

                        # insert data into buffer
                        self.insert(data, rollout)
            else:
                for step in range(self.episode_length):
                    if self.args.sp_check:
                        print('episode = {} step = {}'.format(episode, step))
                    base_ret, q_ret, sp_ret,penalty_ret = self.collect(step)
                    values, actions, action_log_probs, rnn_states, rnn_states_critic,action_env = base_ret

                    # Obser reward and next obs
                        # print(action_env.shape)
                    obs, rewards, dones, infos = self.envs.step(action_env)
                    available_actions = None
                    # if self.args.mpe_share_obs:
                    #     tmp_shape = np.shape(obs)
                    #     share_obs = obs.reshape([tmp_shape[0], 1, -1]).repeat(self.args.n_agents, axis=1)
                   

                    if self.args.reward_scale_const is not None:
                        rewards = rewards / self.args.reward_scale_const

                    obs = self.dict_to_tensor(obs)
                    rewards = self.dict_to_tensor(rewards, False)
                    rewards = np.expand_dims(rewards, -1)
                    base_input = [obs, share_obs, rewards, dones, infos, available_actions, \
                                  values, actions, action_log_probs, \
                                  rnn_states, rnn_states_critic]
                    data = [base_input, q_ret, sp_ret,penalty_ret]


                    # insert data into buffer

                    # insert data into buffer
                    self.insert(data)

            # compute return and update network
            self.compute()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            train_infos = self.train(time_steps = total_num_steps)

            # post process

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(
                        incre_battles_game) > 0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                elif self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info['agent_' + str(agent_id)].keys():
                                idv_rews.append(
                                    info['agent_' + str(agent_id)]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews


                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x * y, list(
                        self.buffer.active_masks.shape))

                self.log_train(train_infos, total_num_steps,episodes)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self,rollout=None):
        # reset env
        obs = self.envs.reset()
        obs = self.dict_to_tensor(obs)
      #last_actions = np.zeros(
          #(self.n_rollout_threads, self.num_agents, flatdim(self.envs.action_space('agent_0')) - 1))
      # replay buffer
        if self.use_centralized_V:
                share_obs = obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(
                    self.num_agents, axis=1)
                #last_actions = last_actions.reshape(self.n_rollout_threads, -1)
                #last_actions = np.expand_dims(last_actions, 1).repeat(
                    #self.num_agents, axis=1)
        else:
                share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        # self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step, rollout=None):
        self.trainer.prep_rollout()

        
        available_action_input = None

        if rollout is None:
            share_obs_input = np.concatenate(self.buffer.share_obs[step])
            # print("self.buffer.obs",self.buffer.obs.shape)
            obs_input = np.concatenate(self.buffer.obs[step])
            # print("obs_input",self.buffer.obs[step].shape)
            # print('obs_input = {}'.format(obs_input.shape))
            # (3,30) (agents,obs_shape)
            rnn_states_input = np.concatenate(self.buffer.rnn_states[step])
            # print('rnn_states_input = {}'.format(rnn_states_input.shape))
            # (3,1,64) (agents,1,hidden_size)
            rnn_states_critic_inputs = np.concatenate(self.buffer.rnn_states_critic[step])
            # print('rnn_states_critic_inputs = {}'.format(rnn_states_critic_inputs.shape))
            # (3,1,64) (agents,1,hidden_size)
            masks_inputs = np.concatenate(self.buffer.masks[step])
            # print('masks_inputs = {}'.format(masks_inputs.shape))
            # (3,1) (agents,1)
            if self.args.use_q or self.args.sp_use_q:
                target_rnn_states_critic_inputs = np.concatenate(self.buffer.target_rnn_states_critic[step])
        else:
            share_obs_input = self.buffer.share_obs[step, rollout]
            obs_input = self.buffer.obs[step, rollout]
            rnn_states_input = self.buffer.rnn_states[step, rollout]
            rnn_states_critic_inputs = self.buffer.rnn_states_critic[step, rollout]
            masks_inputs = self.buffer.masks[step, rollout]
            if self.args.use_q or self.args.sp_use_q:
                target_rnn_states_critic_inputs = self.buffer.target_rnn_states_critic[step, rollout]

        if self.args.use_q or self.args.sp_use_q:
            value, action, action_log_prob, rnn_state, rnn_state_critic, target_value, target_rnn_state_critic \
                = self.trainer.policy.get_actions(share_obs_input,
                                                  obs_input,
                                                  rnn_states_input,
                                                  rnn_states_critic_inputs,
                                                  masks_inputs,available_actions=available_action_input, target=True,
                                                  target_rnn_states_critic=target_rnn_states_critic_inputs)
            if self.args.sp_use_q:
                probs = torch.ones_like(value).to(value.device)
            else:
                probs = self.trainer.policy.get_probs(obs_input, rnn_states_input, masks_inputs,available_actions=available_action_input)
            baseline = value * probs
            baseline = torch.sum(baseline, dim=-1, keepdim=True)
            act_idx = action
            # print('act_idx = {} prev_value = {}'.format(act_idx.shape,value.shape))
            old_values_all = value
            old_probs_all = probs
            if not self.args.sp_use_q:
                value = torch.gather(value, index=act_idx, dim=-1)
                target_value = torch.gather(target_value, index=act_idx, dim=-1)
            # print('latter_value = {}'.format( value.shape))
        else:
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(share_obs_input,
                                                  obs_input,
                                                  rnn_states_input,
                                                  rnn_states_critic_inputs,
                                                  masks_inputs,available_actions=available_action_input)
            # print('value = {} action = {} action_log_prob = {}'.format(value.shape,action.shape,action_log_prob.shape))
        # [self.envs, agents, dim]
        # act_idx = torch.argmax(action,dim = -1 ,keepdim= True)


        # print(action.shape)

        split_num = self.n_rollout_threads if rollout is None else 1
        values = np.array(np.split(_t2n(value), split_num))
        # print(action.shape)
        actions = np.array(np.split(_t2n(action), split_num))
        # print(actions)
        
        action_log_probs = np.array(np.split(_t2n(action_log_prob), split_num))
        rnn_states = np.array(np.split(_t2n(rnn_state), split_num))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), split_num))

        # if self.args.env_name == 'MPE':
        #     if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
        #         for i in range(self.envs.action_space[0].shape):
        #             uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
        #             if i == 0:
        #                 actions_env = uc_actions_env
        #             else:
        #                 actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        #     elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
        #         actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        #     else:
        #         raise NotImplementedError
        # else:
        actions_env = np.clip(actions, -1, 1)
        # actions_env = []
        # for i in range(self.n_rollout_threads):
        #     acts = {}
        #     clipped_actions = np.clip(actions, -1, 1)
        #     for j in range(self.num_agents):
        #         acts['agent_' + str(j)] = np.squeeze(clipped_actions[i, j, :])
        #         actions_env.append(acts)


        if self.args.penalty_method:
            if not self.args.env_name == 'mujoco':
                probs = self.trainer.policy.get_probs(obs_input, rnn_states_input, masks_inputs,available_actions=available_action_input)
                # print('penalty_old_probs = {}'.format(probs.shape))
                probs = np.array(np.split(_t2n(probs), split_num))
                penalty_ret = [probs]
            else:
                penalty_ret = None
        else:
            penalty_ret = None
        if self.args.sp_clip:
            sp_value, sp_prob = [], []
            for i in range(self.args.sp_num):
                value, action, action_log_prob, rnn_state, rnn_state_critic \
                    = self.trainer.policy.get_actions(share_obs_input,
                                                      obs_input,
                                                      rnn_states_input,
                                                      rnn_states_critic_inputs,
                                                      masks_inputs,available_actions=available_action_input)
                if self.args.sp_check:
                    print('sp {}: value = {}, action = {}  action_log_prob = {}'.format(i, value, action,
                                                                                        action_log_prob))
                sp_value.append(value)
                sp_prob.append(action_log_prob)

            sp_value = torch.stack(sp_value, dim=-1)
            sp_prob = torch.stack(sp_prob, dim=-1)
            sp_prob = torch.exp(sp_prob)

            sp_value = np.array(np.split(_t2n(sp_value), split_num))
            sp_prob = np.array(np.split(_t2n(sp_prob), split_num))
        # print('collect:: action_log_probs = {}'.format(action_log_probs.shape))
        ret_list = [values, actions, action_log_probs, rnn_states, rnn_states_critic,actions_env]
        q_ret_list, sp_ret_list = None, None
        if self.args.use_q or self.args.sp_use_q:
            old_values_all = np.array(np.split(_t2n(old_values_all), split_num))
            old_probs_all = np.array(np.split(_t2n(old_probs_all), split_num))
            baselines = np.array(np.split(_t2n(baseline), split_num))
            target_values = np.array(np.split(_t2n(target_value), split_num))
            target_rnn_states_critic = np.array(np.split(_t2n(target_rnn_state_critic), split_num))
            q_ret_list = [target_values, target_rnn_states_critic, baselines, old_probs_all, old_values_all]
        if self.args.sp_clip:
            sp_ret_list = [sp_value, sp_prob]
        return ret_list, q_ret_list, sp_ret_list,penalty_ret

    def insert(self, data, rollout=None):
        ret_list, q_ret_list, sp_ret_list,penalty_ret = data
        obs, share_obs, rewards, dones, infos, available_actions, \
        value \
            , actions, action_log_probs, \
        rnn_states, rnn_states_critic = ret_list
        if q_ret_list is not None:
            target_values, target_rnn_states_critic, baseline, old_probs_all, old_values_all = q_ret_list
        else:
            target_rnn_states_critic, target_values, baseline, old_probs_all, old_values_all = None, None, None, None, None

        if sp_ret_list is not None:
            sp_value, sp_prob = sp_ret_list
        else:
            sp_value, sp_prob = None, None

        if penalty_ret is not None:
            penalty_old_probs = penalty_ret[0]
            # print('penalty_old_probs_insert = {}'.format(penalty_old_probs.shape))
        else:
            penalty_old_probs = None

        if self.args.use_q or self.args.sp_use_q:
            values = target_values
            value_curr = value
        else:
            value_curr = None
            values = value
        dones = [[value for value in d.values()] for d in dones]
        dones = np.array(dones)
        dones_env = np.all(dones, axis=1)

        # print('dones = {} dones_env = {}'.format(dones.shape,dones_env.shape))
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.actor_hidden_size * 2), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        if self.args.use_q:
            target_rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.num_agents, *self.buffer.target_rnn_states_critic.shape[3:]),
                dtype=np.float32)
        if rollout is None:
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        else:
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
            active_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs

        if rollout is None:
            insert_mode = 'multi'
        else:
            insert_mode = 'single'

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks,
                           available_actions, insert_mode=insert_mode, curr_rollout=rollout, \
                           target_rnn_states_critic=target_rnn_states_critic, value_curr=value_curr, \
                           baseline=baseline, old_probs_all=old_probs_all, old_values_all=old_values_all, \
                           sp_value=sp_value, sp_prob=sp_prob,penalty_old_probs=penalty_old_probs)




    def log_train(self, train_infos, total_num_steps,episodes):
        # print('shape = {}'.format(np.shape(self.buffer.rewards[self.buffer.step])))
        # print(self.buffer.rewards)
        if self.all_args.env_name in ['matrix_game', 'mujoco','StarCraft2','MPE']:
            train_infos["average_step_rewards"] = np.sum(self.buffer.rewards[self.buffer.step]) / self.args.n_rollout_threads *self.episode_length
        elif self.all_args.env_name == 'DiffGame':
            train_infos["average_step_rewards"] = np.mean(self.buffer.rewards[self.buffer.step])

        # train_infos["last_episode_sum"] = np.sum(self.buffer.rewards[self.buffer.step])
        print('last_episode_num = {}'.format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            # print('log_train key = {}'.format(k))
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    