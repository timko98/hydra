import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

conv_nr = 0
linear_nr = 0

# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        """ Weight pruning
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        """
        """ Channel pruning without changed score mask
        out = scores.clone()
        kept_weights = torch.topk(torch.linalg.norm(out.reshape(out.shape[0], -1), 1, dim=1),
                                  int(k * out.shape[0])).indices
        out[:] = 0
        out[kept_weights] = 1
        """
        # """ Channel pruning with changed score mask
        out = scores.clone()
        kept_weights = torch.topk(out, k=int(k*out.shape[1]), dim=1).indices
        out = torch.transpose(out, 0,1)
        out[:] = 0
        out[kept_weights] = 1
        out = torch.transpose(out, 0,1)
        # """
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # Weight pruning
        # self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        # Channel Finetuning or Resume Pruning
        self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1],1,1])))
        # Channel Pruning
        # self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0], 1,1,1])))

        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        """ Unstructured comparison
        remaining_weights = int(self.k * len(self.weight.flatten()))
        idx_same_top_weights_scores = list(
            set(torch.topk(self.weight.abs().flatten(), remaining_weights).indices.tolist()).intersection(
                set(torch.topk(self.popup_scores.abs().flatten(), remaining_weights).indices.tolist())))
        num_remaining_weights = len(idx_same_top_weights_scores)
        print(
            f"SubnetConv: Number of same indices for scores and weights that are left after pruning: "
            f"{num_remaining_weights}. These are {float(num_remaining_weights / remaining_weights)} percent of the "
            f"weights kept.")
        """
        """ Structured Comparison
        remaining_filters = int(self.k * self.weight.shape[0])
        idx_same_top_weights_scores = list(set(
            torch.topk(torch.linalg.norm(self.weight.abs().reshape(self.weight.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()).intersection(
            torch.topk(torch.linalg.norm(self.popup_scores.abs().reshape(self.popup_scores.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()))
        num_remaining_filters = len(idx_same_top_weights_scores)
        print(
            f"SubnetConv: Number of same indices for filters that are left after pruning using scores or weights : "
            f"{num_remaining_filters}. These are {float(num_remaining_filters / remaining_filters)} percent of the "
            f"filters kept.")
        """
        """ Channel Prune VGG16
        global conv_nr
        if conv_nr == 13:
            conv_nr = 1
        else:
            conv_nr += 1
        # Get the subnetwork by sorting the scores.
        mask_conv_50 = [1.0, 1.0, 0.984375, 1.0, 1.0, 0.98828125, 0.98046875, 1.0, 0.96875, 0.359375, 0.099609375, 0.1015625, 0.099609375]
        mask_conv_10 = [1.0, 0.5, 0.46875, 0.4921875, 0.484375, 0.4765625, 0.5, 0.5, 0.48242188, 0.05078125, 0.0234375, 0.015625, 0.015625]
        mask_conv_10_new = [1.0, 0.75, 0.640625, 0.703125, 0.671875, 0.14453125, 0.01171875, 0.01171875, 0.01171875, 0.009765625, 0.009765625, 0.009765625, 0.009765625] 
        k = mask_conv_10_new[conv_nr-1]
        adj = GetSubnet.apply(self.popup_scores.abs(), k)
        """
        """
        if conv_nr == 1:
            adj = GetSubnet.apply(self.popup_scores.abs(), 1)
        else:
            adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        """
        # """ WRN-28-4
        global conv_nr
        if conv_nr == 28:
            conv_nr = 1
        else:
            conv_nr += 1
        # """
        # """
        mask_wrn_50 = [1, 0.5, 0.171875, 0.5, 0.5625, 0.5, 0.359375, 0.40625, 0.375, 0.1875, 0.390625, 0.4453125, 0.390625, 0.328125, 0.171875, 0.4765625, 0.3046875, 0.140625, 0.2265625,0.640625, 0.49609375, 0.640625, 0.6015625, 0.6796875, 0.46875, 0.52734375, 0.50390625, 0.48825125]
        mask_wrn_10 = [1.0, 0.0625, 0.078125, 0.0625, 0.09375, 0.046875, 0.109375, 0.046875, 0.0625, 0.0625, 0.0625, 0.203125, 0.0625, 0.0703125, 0.1796875, 0.0234375, 0.09375, 0.1953125, 0.1796875, 0.3359375, 0.3359375, 0.1796875, 0.20703125, 0.015625, 0.015625, 0.015625, 0.015625, 0.015625]
        # Add mask for 0.1 Channel pruning here
        k = mask_wrn_10[conv_nr-1]
        adj = GetSubnet.apply(self.popup_scores.abs(), k)
        #  """
        """
        if conv_nr == 1:
            adj = GetSubnet.apply(self.popup_scores.abs(), 1)
        else:
            adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        # """
        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        # Weight pruning
        # self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        # Channel Finetuning or Resume Pruning
        self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1]])))
        # Channel Pruning
        # self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0],1])))

        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        # self.register_buffer('w', None)

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        """ Unstructured Comparison
        remaining_weights = int(self.k * len(self.weight.flatten()))
        idx_same_top_weights_scores = list(
            set(torch.topk(self.weight.abs().flatten(), remaining_weights).indices.tolist()).intersection(
                set(torch.topk(self.popup_scores.abs().flatten(), remaining_weights).indices.tolist())))
        num_remaining_weights = len(idx_same_top_weights_scores)
        print(
            f"SubnetLinear: Number of same indices for scores and weights that are left after pruning: "
            f"{num_remaining_weights}. These are {float(num_remaining_weights / remaining_weights)} percent of the "
            f"weights kept.")
        """
        """ Structured Comparison
        remaining_filters = int(self.k * self.weight.shape[0])
        idx_same_top_weights_scores = list(set(
            torch.topk(torch.linalg.norm(self.weight.abs().reshape(self.weight.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()).intersection(
            torch.topk(torch.linalg.norm(self.popup_scores.abs().reshape(self.popup_scores.shape[0], -1), 1, dim=1),
                       remaining_filters).indices.tolist()))
        num_remaining_filters = len(idx_same_top_weights_scores)
        print(
            f"SubnetLinear: Number of same indices for filters that are left after pruning using scores or weights : "
            f"{num_remaining_filters}. These are {float(num_remaining_filters / remaining_filters)} percent of the "
            f"filters kept.")
        """
        """ Channel Prune VGG16
        global linear_nr
        if linear_nr == 3:
            linear_nr = 1
        else:
            linear_nr += 1
        # Get the subnetwork by sorting the scores.
        mask_linear_50 = [0.10107422, 0.1015625, 0.1015625]
        mask_linear_10 = [0.016601562, 0.015625, 0.015625]
        mask_linear_10_new = [0.010253906, 0.01171875, 0.01171875]
        
        k = mask_linear_10_new[linear_nr-1]
        adj = GetSubnet.apply(self.popup_scores.abs(), k)
        """
        # Fixed mask WRN Channel Prune 0.5
        # adj = GetSubnet.apply(self.popup_scores.abs(), 0.44140625)
        # Fixed mas WRN Channel Prune 0.1
        adj = GetSubnet.apply(self.popup_scores.abs(), 0.2890625)
        # adj = GetSubnet.apply(self.popup_scores.abs(), None)
        # adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x
