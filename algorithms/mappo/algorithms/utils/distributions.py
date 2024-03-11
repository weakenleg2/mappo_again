import torch
import torch.nn as nn
from torch.distributions import Beta
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)#.sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()

class FixedBeta(Beta):
    def __init__(self, alpha, beta, low, high):
        super().__init__(alpha, beta)
        self.low = low
        self.high = high

    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        return self.low + samples * (self.high - self.low)

    def log_probs(self, actions):
        epsilon = 1e-6
        clipped_actions = torch.clamp(actions, self.low + epsilon, self.high - epsilon)
        scaled_actions = (clipped_actions - self.low) / (self.high - self.low)

        clipped_scaled_actions = torch.clamp(scaled_actions, epsilon, 1 - epsilon)
        return super().log_prob(clipped_scaled_actions)

    def entropy(self):
        entropy_values = super().entropy()
        summed_entropy = entropy_values.sum(-1)
        print(summed_entropy)
        return summed_entropy

    def mode(self):
        return self.low + self.mean * (self.high - self.low)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = init_(nn.Linear(num_inputs, num_outputs))
        #self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x, available_actions=None):
        action_mean = self.fc_mean(x)
        action_logstd = torch.clamp(self.logstd(x), min=-6, max=2)
        return FixedNormal(action_mean, action_logstd.exp())

class DiagBeta(DiagGaussian):
    def __init__(self, num_inputs, num_outputs, low=0, high=1 ,use_orthogonal=True, gain=0.01):
        super(DiagBeta, self).__init__(num_inputs, num_outputs, use_orthogonal, gain)
        self.low = low
        self.high = high

    def forward(self, x, available_actions=None):

        action_alpha = torch.exp(self.fc_mean(x))

        zeros = torch.zeros(action_alpha.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_beta = torch.exp(self.logstd(zeros))

        return FixedBeta(action_alpha, action_beta, self.low, self.high)

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x, available_actions=None):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
