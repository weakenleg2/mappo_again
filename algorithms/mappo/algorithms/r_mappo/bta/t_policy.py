import numpy as np
import time
import math

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from algorithms.mappo.utils.util_tem import get_gard_norm, huber_loss, mse_loss, cal_acyclic_loss, generate_mask_from_order
from algorithms.mappo.utils.valuenorm_bta import ValueNorm
from algorithms.mappo.utils.util_tem import check
from .temporalPolicy import TemporalPolicy
# give up graph training

class T_POLICY():
    def __init__(self,
                 args,
                 policy: TemporalPolicy,
                 agent_id,
                 action_space,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = args.num_agents
        self.agent_id = agent_id
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.entropy_lr = args.entropy_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.skip_connect = args.skip_connect 
        self.use_action_attention = args.use_action_attention

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.gamma = args.gamma
        self.data_chunk_length = args.data_chunk_length
        self.policy_value_loss_coef = args.policy_value_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.target_entropy_discount = args.target_entropy_discount
        self.average_threshold = args.average_threshold
        self.standard_deviation_threshold = args.standard_deviation_threshold
        self.exponential_avg_discount = args.exponential_avg_discount
        self.exponential_var_discount = args.exponential_var_discount
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.automatic_target_entropy_tuning = args.automatic_target_entropy_tuning
        self.avg_entropy = nn.Parameter(torch.zeros(1), requires_grad=False).to(**self.tpdv)
        self.var_entropy = nn.Parameter(torch.zeros(1), requires_grad=False).to(**self.tpdv)
        self.target_entropy = torch.zeros(1).to(self.device)
        self.discrete = False
        self.continuous = False
        if action_space.__class__.__name__ == "Discrete":
            self.discrete = True
        elif action_space.__class__.__name__ == "Box":
            self.continuous = True
        self.entropy_coef = args.entropy_coef
        self.shaped_info_coef = getattr(args, "shaped_info_coef", 0.5)
        self.max_grad_norm = args.max_grad_norm       
        self.inner_max_grad_norm = args.inner_max_grad_norm       
        self.huber_delta = args.huber_delta
        self.share_policy = args.share_policy
        self.threshold = args.threshold

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = args.use_policy_vhead
        self.use_graph = args.use_graph
        self._predict_other_shaped_info = (args.env_name == "Overcooked" and getattr(args, "predict_other_shaped_info", False))
        self._policy_group_normalization = (args.env_name == "Overcooked" and getattr(args, "policy_group_normalization", False))
        self._use_task_v_out = getattr(args, "use_task_v_out", False)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.
        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    # def graph_update(self, sample):
    #     # with torch.autograd.set_detect_anomaly(True):
    #     share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, one_hot_actions_batch, \
    #     value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
    #     adv_targ, available_actions_batch, factor_batch, action_grad,_,_ = sample

    #     old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
    #     adv_targ = check(adv_targ).to(**self.tpdv)
    #     value_preds_batch = check(value_preds_batch).to(**self.tpdv)
    #     return_batch = check(return_batch).to(**self.tpdv)
    #     active_masks_batch = check(active_masks_batch).to(**self.tpdv)

    #     #test
    #     # execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * self.agent_id +
    #     #                                 [torch.zeros(actions_batch.shape[0])] *
    #     #                                 (self.num_agents - self.agent_id), -1).to(**self.tpdv)
    #     if self.skip_connect:
    #         execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * self.agent_id +
    #                                         [torch.zeros(actions_batch.shape[0])] *
    #                                         (self.num_agents - self.agent_id), -1).to(**self.tpdv)
    #     else:
    #         if self.agent_id != 0:
    #             execution_masks_batch = torch.stack([torch.zeros(actions_batch.shape[0])] * (self.agent_id - 1) +
    #                                             [torch.ones(actions_batch.shape[0])] * 1 +
    #                                             [torch.zeros(actions_batch.shape[0])] *
    #                                             (self.num_agents - self.agent_id), -1).to(**self.tpdv)
    #         else:
    #             execution_masks_batch = torch.stack([torch.zeros(actions_batch.shape[0])] * self.num_agents, -1).to(**self.tpdv)

        
    #     # Reshape to do in a single forward pass for all steps
    #     values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
    #                                                                         obs_batch, 
    #                                                                         rnn_states_batch, 
    #                                                                         rnn_states_critic_batch, 
    #                                                                         actions_batch, 
    #                                                                         masks_batch, 
    #                                                                         one_hot_actions_batch,
    #                                                                         execution_masks_batch,
    #                                                                         available_actions_batch,
    #                                                                         active_masks_batch,
    #                                                                         )
    #     # actor update
    #     imp_weights = action_log_probs

    #     surr = imp_weights * adv_targ

    #     if self._use_policy_active_masks:
    #         policy_action_loss = (-torch.sum(surr,
    #                                          dim=-1,
    #                                          keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
    #     else:
    #         policy_action_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()

    #     policy_loss = policy_action_loss

    #     # acyclic loss
    #     acyclic_loss = 0
    #     # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
    #     # for i in range(adjs_batch.shape[0]):
    #     #     adjs_ = adjs_batch[i]
    #     #     acyclic_loss += cal_acyclic_loss(adjs_, self.num_agents)
    #     acyclic_loss = acyclic_loss / adjs_batch.shape[0]

    #     return policy_loss + acyclic_loss
    
    def ppo_update(self, sample, train_id, ordered_vertices, tau=1.0):
        # with torch.autograd.set_detect_anomaly(True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, one_hot_actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, action_grad,_,_,_,_,_,_ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        factor_batch = check(factor_batch).to(**self.tpdv)
        action_grad = check(action_grad).to(**self.tpdv)
        #test
        # execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * self.agent_id +
        #                                 [torch.zeros(actions_batch.shape[0])] *
        #                                 (self.num_agents - self.agent_id), -1).to(**self.tpdv)
        # if agent_order is None:
        # agent_order = torch.stack([torch.randperm(self.num_agents) for _ in range(actions_batch.shape[0])]).to(self.device)
        # else:
        agent_order = torch.stack([check(ordered_vertices) for _ in range(actions_batch.shape[0])]).to(self.device)
        execution_masks_batch = generate_mask_from_order(
            agent_order, ego_exclusive=False).to(
                self.device).float()[:, self.agent_id]  # [bs, n_agents, n_agents]
        # execution_masks_batch = torch.stack([torch.ones(actions_batch.shape[0])] * self.agent_id +
        #                                 [torch.zeros(actions_batch.shape[0])] *
        #                                 (self.num_agents - self.agent_id), -1).to(self.device)
        
        actions = torch.from_numpy(actions_batch).to(self.device)
        old_action_log_probs = old_action_log_probs_batch
        one_hot_actions = one_hot_actions_batch[:,0:self.num_agents]
        # Reshape to do in a single forward pass for all steps
        values, train_actions, action_log_probs, _, dist_entropy, _, _ = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions, 
                                                                            masks_batch, 
                                                                            one_hot_actions,
                                                                            execution_masks_batch,
                                                                            available_actions_batch,
                                                                            active_masks_batch,
                                                                            tau=tau
                                                                            )
        # if self.continuous:
        #     train_actions = torch.exp(action_log_probs) / ((-torch.exp(action_log_probs) * (actions - train_actions.mean) / (train_actions.stddev ** 2 + 1e-5)).detach() + 1e-5)
        # elif self.discrete:
        #     train_actions = torch.exp(action_log_probs) / ((torch.exp(action_log_probs)*(1-torch.exp(action_log_probs))).detach() + 1e-5)
        # actor update
        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs),-1,keepdim=True)
        factor_batch = torch.clamp(
                        factor_batch,
                        1.0 - self.clip_param/2,
                        1.0 + self.clip_param/2,
                    ) 
        surr1 = imp_weights * adv_targ * factor_batch+ \
            imp_weights.detach() * action_grad * train_actions
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ * factor_batch + \
            torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param) * action_grad * train_actions

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # surr1 = (imp_weights * factor_batch + (imp_weights.detach()) * action_grad * train_actions) * adv_targ
        # surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * factor_batch \
        #         + (torch.clamp(imp_weights.detach(), 1.0 - self.clip_param, 1.0 + self.clip_param)) * action_grad * train_actions) * adv_targ

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        
        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # # entropy update
        # if self.automatic_entropy_tuning:
        #     if self.automatic_target_entropy_tuning:
        #         delta = (dist_entropy - self.avg_entropy).detach()
        #         self.avg_entropy.add_(delta * (1.0 - self.exponential_avg_discount))
        #         self.var_entropy.mul_(self.exponential_var_discount).add_((delta ** 2) * (1.0 - self.exponential_var_discount))
        #         if (self.target_entropy - self.average_threshold) < self.avg_entropy < (self.target_entropy + self.average_threshold) and (torch.sqrt(self.var_entropy) < self.standard_deviation_threshold):
        #             self.target_entropy *= self.target_entropy_discount
        # entropy_loss = -(self.log_entropy_coef * (action_log_probs + self.target_entropy).detach()).mean()

        # self.entropy_coef_optim.zero_grad()
        # entropy_loss.backward()
        # self.entropy_coef_optim.step()

        # self.entropy_coef = self.log_entropy_coef.exp()
        # else:
        #     entropy_loss = torch.tensor(0.).to(self.device)
        #     entropy_tlogs = self.entropy_coef # For TensorboardX log

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def compute_advantages(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        return advantages

    def train_adv(self, buffer):
        if self._use_popart or self._use_valuenorm:
            # if self.use_action_attention:
            #     advantages = buffer.rewards + self.gamma * buffer.returns[1:] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            # else:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            # if self.use_action_attention:
            #     advantages = buffer.rewards + self.gamma * buffer.returns[1:] - buffer.value_preds[:-1]
            # else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        # if self.args.env_name != "matrix":
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        return advantages
    
    def train(self, buffer, train_id, train_list, tau=1.0):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        # if self.args.env_name != "matrix":
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = defaultdict(float)

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        if self.use_graph:
            train_info['graphic_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, train_id, train_list, tau)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['ratio'] += imp_weights.mean().item()
                # train_info['target_entropy'] += self.target_entropy.item()
                
                if int(torch.__version__[2]) < 5:
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                else:
                    train_info['actor_grad_norm'] += actor_grad_norm.item()
                    train_info['critic_grad_norm'] += critic_grad_norm.item()
    
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info
    
    # def train_graph(self, buffer, turn_on=True):
    #     if self._use_popart or self._use_valuenorm:
    #         advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
    #     else:
    #         advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
    #     advantages_copy = advantages.copy()
    #     advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
    #     mean_advantages = np.nanmean(advantages_copy)
    #     std_advantages = np.nanstd(advantages_copy)
    #     advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    #     if self._use_recurrent_policy:
    #         data_generator = buffer.graph_recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
    #     elif self._use_naive_recurrent:
    #         data_generator = buffer.graph_naive_recurrent_generator(advantages, self.num_mini_batch)
    #     else:
    #         data_generator = buffer.graph_feed_forward_generator(advantages, self.num_mini_batch)

    #     graph_loss = 0
    #     for sample in data_generator:

    #         loss = self.graph_update(sample)
        #     graph_loss += loss
 
        # return graph_loss

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    
    def to(self, device):
        self.policy.to(device)
