# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PopArt(torch.nn.Module):
    
#     def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        
#         super(PopArt, self).__init__()

#         self.beta = beta
#         self.epsilon = epsilon
#         self.norm_axes = norm_axes
#         self.tpdv = dict(dtype=torch.float32, device=device)

#         self.input_shape = input_shape
#         self.output_shape = output_shape

#         self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
#         self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
        
#         self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
#         self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
#         self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
#         self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             torch.nn.init.uniform_(self.bias, -bound, bound)
#         self.mean.zero_()
#         self.mean_sq.zero_()
#         self.debiasing_term.zero_()

#     def forward(self, input_vector):
#         if type(input_vector) == np.ndarray:
#             input_vector = torch.from_numpy(input_vector)
#         input_vector = input_vector.to(**self.tpdv)

#         return F.linear(input_vector, self.weight, self.bias)
    
#     @torch.no_grad()
#     def update(self, input_vector):
#         if type(input_vector) == np.ndarray:
#             input_vector = torch.from_numpy(input_vector)
#         input_vector = input_vector.to(**self.tpdv)
        
#         old_mean, old_var = self.debiased_mean_var()
#         old_stddev = torch.sqrt(old_var)

#         batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
#         batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

#         self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
#         self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
#         self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

#         self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)
        
#         new_mean, new_var = self.debiased_mean_var()
#         new_stddev = torch.sqrt(new_var)
        
#         self.weight = self.weight * old_stddev / new_stddev
#         self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

#     def debiased_mean_var(self):
#         debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
#         debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
#         debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
#         return debiased_mean, debiased_var

#     def normalize(self, input_vector):
#         if type(input_vector) == np.ndarray:
#             input_vector = torch.from_numpy(input_vector)
#         input_vector = input_vector.to(**self.tpdv)

#         mean, var = self.debiased_mean_var()
#         out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
#         return out

#     def denormalize(self, input_vector):
#         if type(input_vector) == np.ndarray:
#             input_vector = torch.from_numpy(input_vector)
#         input_vector = input_vector.to(**self.tpdv)

#         mean, var = self.debiased_mean_var()
#         out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
#         out = out.cpu().numpy()

#         return out
import numpy as np

import torch
import torch.nn as nn


class PopArt(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, train=True):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        if train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()
            # print('tuple(range(self.norm_axes)) = {}'.format(tuple(range(self.norm_axes))))
            # batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            # batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))
            # print('input_vector_shape = {}'.format(input_vector.shape))
            # print('running_mean_shape = {}'.format(self.running_mean.shape))
            batch_mean = detached_input.mean()
            batch_sq_mean = (detached_input ** 2).mean()


            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()
        
        return out
