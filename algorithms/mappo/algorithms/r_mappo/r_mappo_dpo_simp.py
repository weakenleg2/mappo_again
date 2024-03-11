import numpy as np
import torch
import torch.nn as nn
from algorithms.mappo.utils.util import get_gard_norm, huber_loss, mse_loss
from algorithms.mappo.utils.valuenorm import ValueNorm
from algorithms.mappo.algorithms.utils.util import check
# 几乎没变
def hard_update(target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.value_clip_param = args.clip_param
        # #######
        self.args = args
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        # print(self._use_value_active_masks,self._use_policy_active_masks)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None
        self.beta_kl = self.args.beta_kl

        self.dtar_kl = self.args.dtar_kl
        self.kl_para1 = self.args.kl_para1
        self.kl_para2 = self.args.kl_para2
        self.kl_lower = self.dtar_kl / self.kl_para1 #0.02/1.5 0.013
        self.kl_upper = self.dtar_kl * self.kl_para1 #0.02*1.5 0.03



        self.beta_sqrt_kl = self.args.beta_sqrt_kl


        self.dtar_sqrt_kl = self.args.dtar_sqrt_kl
        self.sqrt_kl_para1 = self.args.sqrt_kl_para1
        self.sqrt_kl_para2 = self.args.sqrt_kl_para2
        self.sqrt_kl_lower = self.dtar_sqrt_kl / self.sqrt_kl_para1
        self.sqrt_kl_upper = self.dtar_sqrt_kl * self.sqrt_kl_para1

        self.para_upper_bound = self.args.para_upper_bound
        self.para_lower_bound = self.args.para_lower_bound
        self.term_kl = None
        self.term_sqrt_kl = None
        self.p_loss_part1 = None
        self.p_loss_part2 = None
        self.d_coeff = None
        self.d_term = None


        self.term1_grad_norm = None
        self.term2_grad_norm = None
        

        self.term_dist = None

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

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        old_log_probs_batch,adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        old_log_probs_batch = check(old_log_probs_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # probs = self.policy.get_probs(obs_batch, rnn_states_batch, masks_batch)
        # print("probs",probs)
        
        term1 = imp_weights * adv_targ
        term2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        # eps_kl = 1e-9
        eps_sqrt = 1e-12
        with torch.no_grad():
            old_dist_ctrl, old_dist_com = self.policy.get_dist(obs_batch, rnn_states_batch, masks_batch,available_actions=None,target=True)
            # print("old_dist_ctrl, old_dist_com",old_dist_ctrl, old_dist_com)
        new_dist_ctrl, new_dist_com = self.policy.get_dist(obs_batch, rnn_states_batch, masks_batch)
        # print(torch.distributions.kl_divergence(old_dist_ctrl,new_dist_ctrl))
        # kl = []
        ctrl_kl = torch.distributions.kl_divergence(old_dist_ctrl,new_dist_ctrl)
        com_kl = torch.distributions.kl_divergence(old_dist_com,new_dist_com)
        com_kl = com_kl.unsqueeze(-1)
        kl = torch.cat((ctrl_kl,com_kl), dim=-1)
        # print("kl",kl)
        sqrt_kl = torch.sqrt(torch.max(kl + eps_sqrt,eps_sqrt * torch.ones_like(kl)))        # print("self._use_policy_active_masks",self._use_policy_active_masks)
        if self._use_policy_active_masks:
            # print("we use policy active masks")
            policy_action_loss = (-torch.sum(torch.min(term1, term2),
                                                     dim=-1,
                                                     keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            term_sqrt_kl = ( torch.sum(sqrt_kl,dim=-1,keepdim=True)* active_masks_batch).sum() / active_masks_batch.sum()
            term_kl  = (torch.sum(kl,dim=-1,keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            self.term_sqrt_kl = term_sqrt_kl
            self.term_kl = term_kl
            sqrt_coeff = torch.tensor(self.beta_sqrt_kl).to(**self.tpdv).detach()
            kl_coeff = torch.tensor(self.beta_kl).to(**self.tpdv).detach()
            # print(term1, sqrt_coeff * term_sqrt_kl, kl_coeff * term_kl)
            # print(term1)
            # print('term_sqrt_kl = {} term_kl = {}'.format(self.term_sqrt_kl,self.term_kl))                                                          
            policy_loss = policy_action_loss + sqrt_coeff * term_sqrt_kl + kl_coeff * term_kl
            policy_loss = policy_loss.mean()
        else:
            policy_loss = term1 + sqrt_coeff * term_sqrt_kl + kl_coeff * term_kl
            policy_loss = policy_loss.mean()

        hard_update(self.policy.target_actor, self.policy.actor)
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            # print("we norm the grad")
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

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self.term_kl is not None:
            print('term_sqrt_kl = {} term_kl = {} old_beta_sqrt_kl = {}, old_beta_kl = {}'.format(self.term_sqrt_kl,
                                                                                                    self.term_kl,
                                                                                                    self.beta_sqrt_kl,
                                                                                                    self.beta_kl))
            print('prev beta_kl = {}'.format(self.beta_kl))

            if self.args.penalty_beta_type == 'adaptive':
                if self.term_kl < self.kl_lower:
                    self.beta_kl /= self.kl_para2
                    # self.beta_kl = np.maximum(self.para_lower_bound,self.beta_kl)
                elif self.term_kl > self.kl_upper:
                    self.beta_kl *= self.kl_para2
                    # self.beta_kl = np.minimum(self.para_upper_bound, self.beta_kl)

            if self.args.penalty_beta_sqrt_type == 'adaptive':
                if self.term_sqrt_kl < self.sqrt_kl_lower:
                    self.beta_sqrt_kl /= self.sqrt_kl_para2
                    # self.beta_sqrt_kl = np.maximum(self.para_lower_bound, self.beta_sqrt_kl)

                elif self.term_sqrt_kl > self.sqrt_kl_upper:
                    self.beta_sqrt_kl *= self.sqrt_kl_para2
            print('before_clip_beta_sqrt_kl = {}, before_clip_beta_kl = {}'.format(self.beta_sqrt_kl, self.beta_kl))

            if self.beta_kl < self.para_lower_bound:
                self.beta_kl = self.para_lower_bound
            if self.beta_kl > self.para_upper_bound:
                self.beta_kl = self.para_upper_bound
            if self.beta_sqrt_kl < self.para_lower_bound:
                self.beta_sqrt_kl = self.para_lower_bound
            if self.beta_sqrt_kl > self.para_upper_bound:
                self.beta_sqrt_kl = self.para_upper_bound

            print('new_beta_sqrt_kl = {}, new_beta_kl = {}'.format(self.beta_sqrt_kl, self.beta_kl))
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        # print(buffer.returns.shape)
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
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
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        # sets the module into training mode. 
        # This is important for certain types of layers like Dropout and BatchNorm,
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        # eval mode
        self.policy.actor.eval()
        self.policy.critic.eval()
