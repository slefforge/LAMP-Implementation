"""Auxiliary utils for implementing pruning strategies
"""

from collections import OrderedDict, defaultdict
from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np


def hook_applyfn(hook, model, forward=False, backward=False):
    """

    [description]

    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]

    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
                not isinstance(module, nn.Sequential)
                and
                not isinstance(module, nn.ModuleList)
                and
                not isinstance(module, nn.ModuleDict)
                and
                not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {name: weights.detach().cpu().numpy().copy()
              for name, weights in model.named_parameters(recurse=recurse) if name == 'weight'}
    return params


def get_activations(model, input):
    activations = OrderedDict()

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        activations[module] = (input[0].detach().cpu().numpy().copy(),
                               output.detach().cpu().numpy().copy(),)

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def get_gradients(model, inputs, outputs):
    # TODO implement using model.register_backward_hook()
    # So it is harder than it seems, the grad_input contains also the gradients
    # with respect to the weights and so far order seems to be (bias, input, weight)
    # which is confusing
    # Moreover, a lot of the time the output activation we are looking for is the
    # one after the ReLU and F.ReLU (or any functional call) will not be called by
    # the forward or backward hook
    # Discussion here
    # https://discuss.pytorch.org/t/how-to-register-hook-function-for-functional-form/25775
    # Best way seems to be monkey patching F.ReLU & other functional ops
    # That'll also help figuring out how to compute a module graph
    pass


def get_param_gradients(model, inputs, outputs, loss_func=None, by_module=True):
    gradients = OrderedDict()

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    training = model.training
    model.train()
    pred = model(inputs)
    loss = loss_func(pred, outputs)
    loss.backward()

    if by_module:
        gradients = defaultdict(OrderedDict)
        for module in model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()

    else:
        gradients = OrderedDict()
        for name, param in model.named_parameters():
            assert name not in gradients
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy().copy()

    model.zero_grad()
    model.train(training)

    return gradients


def fraction_to_keep(compression, model, prunable_modules):
    """ Return fraction of params to keep to achieve desired compression ratio

    Compression = total / ( fraction * prunable + (total-prunable))
    Using algrebra fraction is equal to
    fraction = total/prunable * (1/compression - 1) + 1

    Arguments:
        compression {float} -- Desired overall compression
        model {torch.nn.Module} -- Full model for which to compute the fraction
        prunable_modules {List(torch.nn.Module)} -- Modules that can be pruned in the model.

    Returns:
        {float} -- Fraction of prunable parameters to keep to achieve desired compression
    """
    from ..metrics import model_size
    total_size, _ = model_size(model)
    prunable_size = sum([model_size(m)[0] for m in prunable_modules])
    nonprunable_size = total_size - prunable_size
    fraction = 1 / prunable_size * (total_size / compression - nonprunable_size)
    assert 0 < fraction <= 1, \
        f"Cannot compress to {1 / compression} model with {nonprunable_size / total_size}" + \
        "fraction of unprunable parameters"
    return fraction


def is_classifier(module):
    """Returns true if a module is labeled a classifier"""
    if hasattr(module, 'is_classifier'):
        return module.is_classifier


def torch_conv_layer_to_affine(
        conv: torch.nn.Conv2d, input_size: Tuple[int, int]
) -> torch.nn.Linear:
    with torch.no_grad():
        w, h = input_size

        # Formula from the Torch docs:
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        output_size = [
            (input_size[i] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i]
            + 1
            for i in [0, 1]
        ]

        in_shape = (conv.in_channels, w, h)
        out_shape = (conv.out_channels, output_size[0], output_size[1])

        fc = nn.Linear(in_features=np.product(in_shape), out_features=np.product(out_shape))
        fc.weight.data.fill_(0.0)

        # Output coordinates
        for xo, yo in range2d(output_size[0], output_size[1]):
            # The upper-left corner of the filter in the input tensor
            xi0 = -conv.padding[0] + conv.stride[0] * xo
            yi0 = -conv.padding[1] + conv.stride[1] * yo

            # Position within the filter
            for xd, yd in range2d(conv.kernel_size[0], conv.kernel_size[1]):
                # Output channel
                for co in range(conv.out_channels):
                    fc.bias[enc_tuple((co, xo, yo), out_shape)] = conv.bias[co]
                    for ci in range(conv.in_channels):
                        # Make sure we are within the input image (and not in the padding)
                        if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                            cw = conv.weight[co, ci, xd, yd]
                            # Flatten the weight position to 1d in "canonical ordering",
                            # i.e. guaranteeing that:
                            # FC(img.reshape(-1)) == Conv(img).reshape(-1)
                            fc.weight[
                                enc_tuple((co, xo, yo), out_shape),
                                enc_tuple((ci, xi0 + xd, yi0 + yd), in_shape),
                            ] = cw

    return fc


def range2d(to_a, to_b):
    for a in range(to_a):
        for b in range(to_b):
            yield a, b


def enc_tuple(tup: Tuple, shape: Tuple) -> int:
    res = 0
    coef = 1
    for i in reversed(range(len(shape))):
        assert tup[i] < shape[i]
        res += coef * tup[i]
        coef *= shape[i]

    return res


def dec_tuple(x: int, shape: Tuple) -> Tuple:
    res = []
    for i in reversed(range(len(shape))):
        res.append(x % shape[i])
        x //= shape[i]

    return tuple(reversed(res))


class CollectConv2dSizes():
    """class used to find th"""
    def __init__(self):
        self.size_list = []

    def __call__(self, m, i, o):
        self.size_list.append(i[0].shape)

    def clear(self):
        self.size_list = []

def get_cnn_input_sizes(model,inputs) -> List[Tuple]:
    j = CollectConv2dSizes()
    for layers in model.modules():
        if isinstance(layers,nn.Conv2d):
            layers.register_forward_hook(j)
    model(inputs)
    return j.size_list

if __name__ == "__main__":
    layer = torch.nn.Conv2d(in_channels=3, out_channels= 10, kernel_size=(3,3))
    input_size = (28,28)
    new_layer = torch_conv_layer_to_affine(layer, input_size)

