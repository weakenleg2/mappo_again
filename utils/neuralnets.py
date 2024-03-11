import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=256, nonlin=F.relu,
                 constrain_out=True, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc_in.weight.data.uniform_(-3e-3, 3e-3)
            self.fc_hidden.weight.data.uniform_(-3e-3, 3e-3)
            self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        #elif discrete_action:
            #self.out_fn = nn.Softmax(dim=-1)
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        x = self.nonlin(self.fc_in(self.in_fn(X)))
        x = self.nonlin(self.fc_hidden(x))
        out = self.out_fn(self.fc_out(x))
        return out

if __name__ == '__main__':
  nn = MLPNetwork(1, 1, 128)
  res = nn(torch.tensor([1.0]))
  print(res)