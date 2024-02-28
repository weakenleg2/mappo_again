import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.mappo.algorithms.bta_utils.util import init, check
from algorithms.mappo.algorithms.bta_utils.cnn import CNNBase
from algorithms.mappo.algorithms.bta_utils.mlp import MLPBase, MLPLayer
from algorithms.mappo.algorithms.bta_utils.rnn import RNNLayer
from algorithms.mappo.algorithms.bta_utils.act import ACTLayer
from algorithms.mappo.algorithms.bta_utils.popart import PopArt
# from bta.utils.util import get_shape_from_obs_space
from gymnasium.spaces.utils import flatdim

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, agent_id, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.actor_hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        # self._use_influence_policy = args.use_influence_policy
        # self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart 
        self._recurrent_N = args.recurrent_N 
        self.use_action_attention = args.use_action_attention
        self.agent_id = agent_id 
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = (flatdim(obs_space),)
        self.whole_cfg = None

        base = MLPBase

        self.base_ctrl = base(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        self.base_com = base(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)

        # 为什么？
        input_size = self.base_ctrl.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.ctrl_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.com_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            # self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        

        self.abs_size = input_size

        if action_space.__class__.__name__ == "Discrete":
            self.action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
        else:
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_dim = continous_dim + discrete_dim

        # input_size += args.num_agents * self.action_dim + args.num_agents
        if self.use_action_attention:
            self.action_base_ctrl = MLPBase(args, [args.num_agents], use_attn_internal=args.use_attn_internal, use_cat_self=True)
            self.action_base_com = MLPBase(args, [args.num_agents], use_attn_internal=args.use_attn_internal, use_cat_self=True)

        else:
            self.action_base_ctrl = MLPBase(args, [args.num_agents * action_space[0] + args.num_agents], use_attn_internal=args.use_attn_internal, use_cat_self=True)
            self.action_base_com = MLPBase(args, [args.num_agents * action_space[1] + args.num_agents], use_attn_internal=args.use_attn_internal, use_cat_self=True)

        self.feature_norm = nn.LayerNorm(input_size*2)

        self.num_agents = args.num_agents

        # self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain, self.use_action_attention)
        self.act_ctrl = ACTLayer(action_space[0], self.hidden_size, self._use_orthogonal, self._gain, self.use_action_attention)
        self.act_com = ACTLayer(action_space[1], self.hidden_size, self._use_orthogonal, self._gain, self.use_action_attention)
        self.to(device)

    def forward(self, obs, rnn_states, masks, onehot_action, execution_mask, available_actions=None, deterministic=False, tau=1.0):        
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states = rnn_states.split(int(self.hidden_size), dim=-1)
        masks = check(masks).to(**self.tpdv)
        onehot_action = check(onehot_action).to(**self.tpdv)
        execution_mask = check(execution_mask).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        control_features = self.base_ctrl(obs)
        communication_features = self.base_com(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            control_features, ctrl_rnn_states = self.ctrl_rnn(control_features, rnn_states[0], masks)
            communication_features, com_rnn_states = self.com_rnn(communication_features, rnn_states[1], masks)
            rnn_states = torch.cat((ctrl_rnn_states, com_rnn_states), dim=-1)

        

        masked_actions = (onehot_action * execution_mask.unsqueeze(-1)).view(*onehot_action.shape[:-2], -1)
        # actor_features = torch.cat([actor_features, masked_actions.view(*masked_actions.shape[:-2], -1)], dim=1)

        id_feat_ctrl = torch.eye(self.args.num_agents)[self.agent_id].unsqueeze(0).repeat(control_features.shape[0], 1).to(control_features.device)
        id_feat_com = torch.eye(self.args.num_agents)[self.agent_id].unsqueeze(0).repeat(communication_features.shape[0], 1).to(communication_features.device)

        if self.use_action_attention:
            actor_features_ctrl = control_features + self.action_base_ctrl(id_feat_ctrl)
            actor_features_com = communication_features + self.action_base_ctrl(id_feat_com)

        else:
            actor_features_ctrl = control_features + self.action_base_ctrl(torch.cat([masked_actions, id_feat_ctrl], dim=1))
            actor_features_com = communication_features + self.action_base_com(torch.cat([masked_actions, id_feat_com], dim=1))
        actor_features = torch.cat((actor_features_com,actor_features_ctrl),dim=-1)
        actor_features = self.feature_norm(actor_features)
        obs_feat = actor_features.clone() 
        ctrl_actions, ctrl_action_log_probs, dist_entropy_ctrl, logits_ctrl = self.act_ctrl(actor_features_ctrl, available_actions, deterministic,tau=tau)
        com_actions, com_action_log_probs, dist_entropy_com, logits_com = self.act_com(actor_features_com, available_actions, deterministic,tau=tau)
        # ctrl_actions, ctrl_action_log_probs, dist_entropy_ctrl = self.act_ctrl(actor_features_ctrl, available_actions, deterministic,tau=tau)
        # com_actions, com_action_log_probs, dist_entropy_com = self.act_com(actor_features_com, available_actions, deterministic,tau=tau)
        print(com_actions)
        actions = torch.cat((ctrl_actions, com_actions), dim=-1)
        action_log_probs = torch.cat((ctrl_action_log_probs, com_action_log_probs), dim=-1)
        # print("logits_ctr",logits_ctrl)
        # print("logits_com",logits_com)

        logits = torch.cat((logits_ctrl.loc, logits_com), dim=-1)
        # print(dist_entropy_ctrl.shape, dist_entropy_com.shape)
        dist_entropy = torch.cat((dist_entropy_ctrl, dist_entropy_com.unsqueeze(-1)), dim=-1)

        return actions, action_log_probs, rnn_states, logits, dist_entropy, obs_feat
    
    def evaluate_actions(self, obs, rnn_states, action, masks, onehot_action, execution_mask, available_actions=None, active_masks=None, tau=1.0, kl=False, joint_actions=None):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states = rnn_states.split(int(self.hidden_size), dim=-1)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        onehot_action = check(onehot_action).to(**self.tpdv)
        execution_mask = check(execution_mask).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if joint_actions is not None:
            joint_actions = check(joint_actions).to(**self.tpdv)
        
        actor_control_features = self.base_ctrl(obs)
        actor_communication_features = self.base_com(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            control_features, ctrl_rnn_states = self.ctrl_rnn(actor_control_features, rnn_states[0], masks)
            communication_features, com_rnn_states = self.com_rnn(actor_communication_features, rnn_states[1], masks)
            rnn_states = torch.cat((ctrl_rnn_states, com_rnn_states), dim=-1)

        masked_actions = (onehot_action * execution_mask.unsqueeze(-1)).view(*onehot_action.shape[:-2], -1)
        # actor_features = torch.cat([actor_features, masked_actions.view(*masked_actions.shape[:-2], -1)], dim=1)

        id_feat_ctrl = torch.eye(self.args.num_agents)[self.agent_id].unsqueeze(0).repeat(control_features.shape[0], 1).to(control_features.device)
        id_feat_com = torch.eye(self.args.num_agents)[self.agent_id].unsqueeze(0).repeat(communication_features.shape[0], 1).to(communication_features.device)

        if self.use_action_attention:
            actor_features_ctrl = control_features + self.action_base_ctrl(id_feat_ctrl)
            actor_features_com = communication_features + self.action_base_ctrl(id_feat_com)

        else:
            actor_features_ctrl = control_features + self.action_base_ctrl(torch.cat([masked_actions, id_feat_ctrl], dim=1))
            actor_features_com = communication_features + self.action_base_com(torch.cat([masked_actions, id_feat_com], dim=1))
        actor_features = torch.cat((actor_features_com,actor_features_ctrl),dim=-1)
        actor_features = self.feature_norm(actor_features)
        obs_feat = actor_features.clone() 

        # actor_features = torch.cat([actor_features, id_feat], dim=1)
        train_actions_ctrl, action_log_probs_ctrl, action_log_probs_kl_ctrl, dist_entropy_ctrl, logits_ctrl = self.act_ctrl.evaluate_actions(actor_features_ctrl,
                                                                action[:, 0:-1], available_actions,
                                                                active_masks=
                                                                active_masks if self._use_policy_active_masks
                                                                else None,rsample=True, tau=tau, kl=kl, joint_actions=joint_actions)

        train_actions_com, action_log_probs_com, action_log_probs_kl_com, dist_entropy_com, logits_com = self.act_com.evaluate_actions(actor_features_com,
                                                                action[:, -1:], available_actions,
                                                                active_masks=
                                                                active_masks if self._use_policy_active_masks
                                                                else None,rsample=True, tau=tau, kl=kl, joint_actions=joint_actions)

        action_log_probs = torch.cat((action_log_probs_ctrl, action_log_probs_com), dim=-1)
        dist_entropy = dist_entropy_ctrl+dist_entropy_com
        train_actions = torch.cat((train_actions_ctrl, train_actions_com), dim=-1)
        action_log_probs_kl = torch.cat((action_log_probs_kl_ctrl, action_log_probs_kl_com), dim=-1)
        logits = torch.cat((logits_ctrl, logits_com), dim=-1)
        return train_actions, action_log_probs, action_log_probs_kl, dist_entropy, logits, obs_feat

class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, action_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.critic_hidden_size
        self._use_orthogonal = args.use_orthogonal  
        self._activation_id = args.activation_id     
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_popart = args.use_popart
        # self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self.num_agents = args.num_agents
        self._num_v_out = getattr(args, "num_v_out", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = (flatdim(share_obs_space),)
        self.whole_cfg = None

        
    
        base = MLPBase
        self.base = base(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.abs_size = input_size
        if self._use_popart:
            self.v_out = init_(PopArt(input_size, self._num_v_out, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, self._num_v_out))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks, task_id=None):
        
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        
        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        
        state_feat = critic_features
        
        values = self.v_out(critic_features)

        return values, rnn_states, state_feat