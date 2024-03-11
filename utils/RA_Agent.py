import torch
import copy
from torch import nn
from torch.optim import Adam
from utils.misc import soft_update
from utils.neuralnets import MLPNetwork

class RA_Agent(nn.Module):
  def __init__(self, in_dim, out_dim, critic_in,
              n_options=2, hidden_dim=128, 
              discrete_action=False, lr=1e-2, 
              options_policy = None,
              control_policy = None,
              ):

    super(RA_Agent, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.critic_in = critic_in

    assert(options_policy is not None)
    assert(control_policy is not None)

    self.options_policy = options_policy
    self.control_policy = control_policy

  def update_target(self, tau = 0.01):
    soft_update(self.target_options_policy, self.options_policy, tau)
    soft_update(self.target_control_policy, self.control_policy, tau)
    soft_update(self.target_critic, self.critic, tau)

  def step(self, state, explore=True):
    """
      Take a step forward in environment for a minibatch of observations
      Inputs:
          obs (PyTorch Variable): Observations for this agent
      Outputs:
          action (PyTorch Variable): Raw action output from policy (no exploration)
      """
    actions = self.actions(state)
    return actions

  def _merge_actions(self, actions, options):
    return torch.cat((actions, options), dim=-1) 

  def actions(self, state):
    actions = self.control_policy(state)
    options = self.options_policy(state)
    return self._merge_actions(actions, options)

  def target_actions(self, state): 
    actions = self.target_control_policy(state)
    options = self.target_options_policy(state)
    return self._merge_actions(actions, options)

  def option(self, state):
    return self.options_policy(state)

  def value(self, X):
    return self.state_option_value(X)

  def get_params(self):
      return {'control_policy': self.control_policy.state_dict(),
              'options_policy': self.options_policy.state_dict(),
              'critic': self.critic.state_dict(),
              'control_optimizer': self.control_optimizer.state_dict(),
              'options_optimizer': self.options_optimizer.state_dict(),
              'critic_optimizer': self.critic_optimizer.state_dict()}

  def load_params(self, params, device):
      self.control_policy.load_state_dict(params['control_policy'])
      self.options_policy.load_state_dict(params['options_policy'])

      self.options_policy.to(device)
      self.control_policy.to(device)

if __name__ == '__main__':
  print("Testing RA_Agent")

  agent = RA_Agent(2, 1, 2)
  actions = agent.target_actions(torch.tensor([[1, 2], [3, 4]], dtype=torch.float))

