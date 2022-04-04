from numpy import ndarray
from torch import Tensor
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from shrinkbench.pruning import VisionPruning
from shrinkbench.strategies import map_importances, flatten_importances, fraction_threshold, importance_masks
import torch




# noinspection PyMethodOverriding

class Snip(VisionPruning):
    def __init__(self, model, loss, dataloader, device,  inputs, outputs, compression=0.0):
        super().__init__(model, inputs, outputs, compression)
        self.model = model
        self.loss = loss
        self.dataloader = dataloader
        self.device = device

    def model_masks(self, model, loss, dataloader, device):
        importances = self.snip_importances()
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

    def snip_importances(self, model, loss, dataloader, device):
        masked_params = self.masked_parameters(model)

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            self.loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


    def prunable(self, module, batchnorm, residual):
        r"""Returns boolean whether a module is prunable.
        """
        isprunable = isinstance(module, (Linear, Conv2d))
        if batchnorm:
            isprunable |= isinstance(module, (BatchNorm1d, BatchNorm2d))
        if residual:
            isprunable |= isinstance(module, (Identity1d, Identity2d))
        return isprunable

    def masks(self, module):
        r"""Returns an iterator over modules masks, yielding the mask.
        """
        for name, buf in module.named_buffers():
            if "mask" in name:
                yield buf

    def masked_parameters(self, model, bias=False, batchnorm=False, residual=False):
        r"""Returns an iterator over models prunable parameters, yielding both the
        mask and parameter tensors.
        """
        for module in filter(lambda p: self.prunable(p, batchnorm, residual), model.modules()):
            for mask, param in zip(self.masks(module), module.parameters(recurse=False)):
                if param is not module.bias or bias is True:
                    yield mask, param
        
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)        
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W
