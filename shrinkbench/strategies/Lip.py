from typing import Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)
from ..pruning.utils import is_classifier, get_params, torch_conv_layer_to_affine


class LipschitzUnstruc(VisionPruning):

    def model_masks(self, ):
        """method used to get mask"""
        param_dict: Dict[Module, Dict[str, Tensor]] = self.params(only_prunable=True)

    def get_classifier(self) -> Dict[Module, Dict[str, Tensor]]:
        """returns the classifier in the {module: {"weight":Tensor}} format"""

        temp_list = [module for module in self.model.modules() if is_classifier(module)]
        if len(temp_list) > 1:
            raise Exception("More than one classifier")
        else:
            classifier = temp_list[0]
        return {classifier: get_params(classifier)}

    def find_param_dict(self, param_dict) -> Dict[Module, Dict[str, Tensor]]:
        final_dict = {}
        cnn_num = 0
        for layers in param_dict.keys():
            if isinstance(layers, nn.Conv2d):
                temp_layer = torch_conv_layer_to_affine(layers, (
                self.cnn_input_sizes[cnn_num][-2], self.cnn_input_sizes[cnn_num][-1]))
                temp_dict = get_params(temp_layer)
                cnn_num += 1
                final_dict[temp_layer] = temp_dict
            else:
                final_dict[layers] = param_dict[layers]
        return final_dict
