import math
import numpy as np
import os
import torch.nn as nn
import torch
from gymnasium.spaces.utils import flatdim
from algorithms.mappo.algorithms.utils.util import init, check
from algorithms.mappo.algorithms.utils.cnn import CNNBase
from algorithms.mappo.algorithms.utils.mlp import MLPBase
from algorithms.mappo.algorithms.utils.rnn import RNNLayer
from algorithms.mappo.algorithms.utils.act import ACTLayer
from algorithms.mappo.algorithms.utils.popart import PopArt
# main change here is 2 actors, one for control one for comm, and flatdim is for obs

class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.actor_hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = (flatdim(obs_space),)
        #base = CNNBase if len(obs_shape) == 3 else MLPBase
        
        #Hack since we're only using MPE
        base = MLPBase
        self.base_ctrl = base(args, self.hidden_size, obs_shape,None,None)
        # print(obs_shape)
        self.base_com = base(args, self.hidden_size, obs_shape,None,None)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.ctrl_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.com_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act_ctrl = ACTLayer(action_space[0], self.hidden_size, self._use_orthogonal, self._gain)
        self.act_com = ACTLayer(action_space[1], self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states = rnn_states.split(int(self.hidden_size), dim=-1)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        # print(obs.shape)

        control_features = self.base_ctrl(obs)
        communication_features = self.base_com(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            control_features, ctrl_rnn_states = self.ctrl_rnn(control_features, rnn_states[0], masks)
            communication_features, com_rnn_states = self.com_rnn(communication_features, rnn_states[1], masks)
            rnn_states = torch.cat((ctrl_rnn_states, com_rnn_states), dim=-1)
        else:
            rnn_states = torch.cat((rnn_states[0], rnn_states[1]), dim=-1)
        
        ctrl_actions, ctrl_action_log_probs = self.act_ctrl(control_features, available_actions, deterministic)
        com_actions, com_action_log_probs = self.act_com(communication_features, available_actions, deterministic)
        
        actions = torch.cat((ctrl_actions, com_actions), dim=-1)
        # print(actions.shape)
        action_log_probs = torch.cat((ctrl_action_log_probs, com_action_log_probs), dim=-1)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states = rnn_states.split(int(self.hidden_size), dim=-1)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        control_features = self.base_ctrl(obs)
        communication_features = self.base_com(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            control_features, ctrl_rnn_states = self.ctrl_rnn(control_features, rnn_states[0], masks)
            communication_features, com_rnn_states = self.com_rnn(communication_features, rnn_states[1], masks)
            rnn_states = torch.cat((ctrl_rnn_states, com_rnn_states), dim=-1)
        else:
            rnn_states = torch.cat((rnn_states[0], rnn_states[1]), dim=-1)

        
        control_log_probs, control_dist_entropy = self.act_ctrl.evaluate_actions(control_features,
                                                                action[:, 0:-1], available_actions,
                                                                active_masks=
                                                                active_masks if self._use_policy_active_masks
                                                                else None)

        communication_log_probs, communication_dist_entropy = self.act_com.evaluate_actions(communication_features,
                                                                action[:, -1:], available_actions,
                                                                active_masks=
                                                                active_masks if self._use_policy_active_masks
                                                                else None)

        action_log_probs = torch.cat((control_log_probs, communication_log_probs), dim=-1)
        dist_entropy = control_dist_entropy + communication_dist_entropy

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.critic_hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        # print("cent_obs_space1",cent_obs_space)
        cent_obs_shape = (flatdim(cent_obs_space),)
        # print("cent_obs_space2",cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, self.hidden_size, cent_obs_shape,None,None)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
