from functools import partial
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn.init import constant_, xavier_uniform_


class FeatureProcessor(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projection(x)


zeros_initializer = partial(constant_, val=0.0)


class Dense(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, activation=None, weight_init=xavier_uniform_, bias_init=zeros_initializer, norm=None, gain=None):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super().__init__(in_features, out_features, bias)
        self.activation = activation() if inspect.isclass(activation) else activation
        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        output = super().forward(inputs)
        if self.norm is not None:
            output = self.norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class MLP1(nn.Module):

    def __init__(self, hidden_dims, bias=True, activation=None, last_activation=None, weight_init=xavier_uniform_, bias_init=zeros_initializer, norm=""):
        super().__init__()
        dense = partial(Dense, bias=bias, weight_init=weight_init, bias_init=bias_init)
        layers = [
            dense(hidden_dims[index], hidden_dims[index + 1], activation=activation, norm=norm)
            for index in range(len(hidden_dims) - 2)
        ]
        layers.append(dense(hidden_dims[-2], hidden_dims[-1], activation=last_activation))
        self.dense_layers = nn.ModuleList(layers)
        self.layers = nn.Sequential(*self.dense_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dense_layers:
            layer.reset_parameters()

    def forward(self, x):
        return self.layers(x)


class CosineCutoff(nn.Module):

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff.item() if isinstance(cutoff, torch.Tensor) else cutoff

    def forward(self, distances):
        values = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return values * (distances < self.cutoff).float()


class NodeInit(MessagePassing):

    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100, activation=F.silu, proj_ln="", last_activation=False, weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_, concat=False):
        super().__init__(aggr="add")
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        first_channel = hidden_channels[0]
        last_channel = hidden_channels[-1]
        self.concat = concat
        self.fc1_stru = nn.Linear(1536, 512)
        self.fc2_stru = nn.Linear(512, last_channel)
        self.silu = nn.SiLU()
        if concat:
            self.embedding_src = nn.Embedding(max_z, first_channel)
            self.distance_proj = MLP1([num_rbf + 2 * first_channel] + hidden_channels, activation=activation, norm=proj_ln, weight_init=weight_init, bias_init=bias_init, last_activation=activation if last_activation else None)
        else:
            self.distance_proj = MLP1([num_rbf, last_channel], activation=None, norm="", weight_init=weight_init, bias_init=bias_init, last_activation=None)
            self.combine = MLP1([2 * last_channel] + hidden_channels, activation=activation, norm=proj_ln, weight_init=weight_init, bias_init=bias_init, last_activation=activation if last_activation else None)
        self.cutoff = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        if self.concat:
            self.embedding_src.reset_parameters()
        self.distance_proj.reset_parameters()
        if not self.concat:
            self.combine.reset_parameters()

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]
        neighbors = self.fc2_stru(self.silu(self.fc1_stru(z)))
        if self.concat:
            source = self.embedding_src(z)
            weights = edge_attr
        else:
            source = neighbors
            weights = self.distance_proj(edge_attr) * self.cutoff(edge_weight).view(-1, 1)
        neighbors = self.propagate(edge_index, x=neighbors, s=source, W=weights, size=None)
        if self.concat:
            return x + neighbors
        return self.combine(torch.cat([x, neighbors], dim=1))

    def message(self, s_i, x_j, W):
        if self.concat:
            return self.distance_proj(torch.cat([W, x_j, s_i], dim=1))
        return x_j * W
