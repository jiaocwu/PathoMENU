
import numpy as np

import math
from math import pi
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import Embedding

from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from models.ops import TensorInit, TensorLayerNorm
from torch_scatter import scatter, scatter_add
from models.common_modules import NodeInit


def nan_to_num(vec, num=0.0):
    idx = torch.isnan(vec)
    vec[idx] = num
    return vec


def _normalize(vec, dim=-1):
    return nan_to_num(
        torch.div(vec, torch.norm(vec, dim=dim, keepdim=True)))


def swish(x):
    return x * torch.sigmoid(x)


class rbf_emb(nn.Module):
    def __init__(self, num_rbf, soft_cutoff_upper, rbf_trainable=False):
        super().__init__()
        self.soft_cutoff_upper = soft_cutoff_upper
        self.soft_cutoff_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        soft_cutoff = 0.5 * \
                      (torch.cos(dist * pi / self.soft_cutoff_upper) + 1.0)
        soft_cutoff = soft_cutoff * (dist < self.soft_cutoff_upper).float()
        return soft_cutoff * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    def __init__(self, input_dim,hid_dim):
        super(NeighborEmb, self).__init__(aggr='add')
        self.fc1_stru = nn.Linear(input_dim, 512)
        self.fc2_stru = nn.Linear(512, hid_dim)
        self.silu = nn.SiLU()
        self.hid_dim = hid_dim

    def forward(self, z, s, edge_index, embs):
        s_neighbors =  self.fc2_stru(self.silu(self.fc1_stru(z)))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class S_vector(MessagePassing):
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU())

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    def __init__(
            self,
            hidden_channels,
            num_radial,
            last_layer,
            num_heads,
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.num_heads = num_heads
        self.last_layer = last_layer

        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.silu = nn.SiLU()

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 5),
        )



        if not self.last_layer:
            self.t_src_proj = nn.Linear(hidden_channels, hidden_channels , bias=False)
            self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels , bias=False)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels , bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels , bias=False)
            self.f_proj = nn.Linear(self.hidden_channels, hidden_channels )




        self.inv_proj_k = nn.Sequential(
            nn.Linear(self.hidden_channels , self.hidden_channels * 3), nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 5), )



        self.inv_sqrt_3 = 1 / math.sqrt(5.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)


        self.inv_proj_k[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.inv_proj_k[0].weight)
        self.inv_proj_k[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.inv_proj_k[2].weight)


        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)

            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)
            nn.init.xavier_uniform_(self.t_trg_proj.weight)
            nn.init.xavier_uniform_(self.t_src_proj.weight)


    def forward(self, x, vec,vec2, edge_index, weight, edge_diff, edge_vector,node_frame):
        xh = self.x_proj(x)
        rbfh = self.inv_proj_k(weight)
        row,col = edge_index

        dx, dvec,dvec2 = self.propagate(
            edge_index,
            x=xh,
            vec=vec,
            vec_2 = vec2,
            rbfh_ij=rbfh,
            r_1_ij=edge_vector,
            r_2_ij=edge_diff,
            target_index=col,
            node_frame=node_frame,
            size=None,
        )

        vec = vec + dvec
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, edge_vector=edge_vector, edge_sca=weight)
            return dx, dvec,dvec2, df_ij
        else:
            return dx, dvec, dvec2, None

    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def edge_update(self, vec_i, vec_j, edge_vector, edge_sca):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), edge_vector)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -edge_vector)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.silu(self.f_proj(edge_sca)) * w_dot
        return df_ij

    def message(self, x_j, vec_j,vec_2_j, rbfh_ij, r_1_ij,r_2_ij,target_index,node_frame_i,node_frame_j):
        trans_frame_ij_expanded = torch.matmul(node_frame_j, node_frame_i.transpose(1, 2))
        v_vec_j_mm = torch.matmul(trans_frame_ij_expanded.transpose(1, 2), vec_2_j)

        xh_j = (x_j * rbfh_ij)

        x, xh2, xh3, xh4, xh5 = torch.split(xh_j, self.hidden_channels, dim=-1)
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_1_ij.unsqueeze(2)

        r_ji_mm = torch.matmul(trans_frame_ij_expanded.transpose(1, 2), xh5.unsqueeze(1) * r_2_ij.unsqueeze(2))
        vec2 = v_vec_j_mm * xh4.unsqueeze(1) + r_ji_mm

        return x, vec, vec2

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec,vec2 = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        vec2 = scatter(vec2, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec, vec2

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs





class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.equi_proj = nn.Linear(
            hidden_channels, hidden_channels * 3, bias=False
        )
        self.equi_proj_2 = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xequi_proj = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 5),
        )
        self.vec_puter = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
        )

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.equi_proj.weight)
        nn.init.xavier_uniform_(self.equi_proj_2.weight)

        nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.fill_(0)

    def forward(self, x, vec_1, vec_2):

        vec_1 = self.equi_proj(vec_1)
        vec_2 = self.equi_proj_2(vec_2)
        vec_1_1, vec_1_2, vec_1_3 = torch.split(
            vec_1, self.hidden_channels, dim=-1
        )
        vec_2_1, vec_2_2= torch.split(
            vec_2, self.hidden_channels, dim=-1
        )

        vec_1_sca = torch.norm(vec_1_1, dim=-2, keepdim=False)
        vec_2_sca = torch.norm(vec_2_1, dim=-2, keepdim=False)

        vec_outer = torch.einsum('eia,eja->eija', vec_1_2, vec_1_3)
        n, i, j, a = vec_outer.shape
        vec_1_agg = vec_outer.view(n, i * j, a).mean(1)

        vec_2_agg = vec_2_2.mean(1)

        x_vec_h = self.xequi_proj(
            torch.cat(
                [x, vec_1_sca, vec_2_sca], dim=-1
            )
        )

        xvec1, xvec2, xvec3, xvec4, xvec5 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = (xvec1 + vec_1_agg * xvec2 + vec_2_agg * xvec3)
        dvec_1 = xvec4.unsqueeze(1) * (vec_1_3)
        dvec_2 = xvec5.unsqueeze(1) * vec_2_2
        return dx, dvec_1, dvec_2




class aggregate_pos(MessagePassing):

    def __init__(self, aggr='mean'):
        super(aggregate_pos, self).__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


class EquiOutput(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


class GatedEquivariantBlock(nn.Module):

    def __init__(
            self,
            hidden_channels,
            out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


def update_features_vectorized(
        dense_adj_tensor: torch.Tensor,
        feature_matrix: torch.Tensor,
        nodes_per_graph: torch.Tensor
) -> torch.Tensor:
    device = feature_matrix.device
    batch_size, max_nodes, _ = dense_adj_tensor.shape
    total_nodes = feature_matrix.shape[0]


    node_range = torch.arange(max_nodes, device=device)
    mask = node_range < nodes_per_graph.unsqueeze(1)
    padding_mask = mask.unsqueeze(2) & mask.unsqueeze(1)

    dense_adj_tensor = dense_adj_tensor * padding_mask

    sparse_adj = dense_adj_tensor.to_sparse_coo()
    batch_indices, row_indices, col_indices = sparse_adj.indices()


    cumulative_nodes = torch.cat([torch.tensor([0], device=device), nodes_per_graph.cumsum(dim=0)])
    offsets = cumulative_nodes[batch_indices]
    row = row_indices + offsets
    col = col_indices + offsets


    num_neighbors = scatter_add(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=total_nodes)

    has_neighbors_mask = num_neighbors > 0
    updated_feature_matrix = feature_matrix.clone()

    if has_neighbors_mask.any():
        edge_weights = 1.0 / num_neighbors[row]
        edge_weights[torch.isinf(edge_weights)] = 0

        weighted_features = feature_matrix[col] * edge_weights.unsqueeze(-1)

        aggregated_features = torch.zeros_like(feature_matrix)
        scatter_add(weighted_features.to(aggregated_features.dtype), row, out=aggregated_features, dim=0)

        updated_feature_matrix[has_neighbors_mask] = aggregated_features[has_neighbors_mask]

    return updated_feature_matrix

class PathoMENUStructureEncoder(torch.nn.Module):

    def __init__(
            self, pos_require_grad=False, cutoff=8.0, num_layers=2, input_dim=1024,
            hidden_channels=128, num_radial=64, y_mean=0, y_std=1, lmax=2, num_heads=8,
            trainable_vecnorm=False, readout="mean", dropout=0.4, **kwargs):
        super().__init__()
        self.y_std = y_std
        self.y_mean = y_mean
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.pos_require_grad = pos_require_grad
        self.output_dim = hidden_channels

        self.z_emb = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_channels))
        self.radial_emb = rbf_emb(num_radial, self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        self.neighbor_emb = NeighborEmb(input_dim,hidden_channels)


        self.neighbor_embedding = NodeInit([hidden_channels // 2, hidden_channels], num_radial, self.cutoff, max_z=100, concat=False,
                                           proj_ln='layer')

        self.tensor_init = TensorInit(l=lmax)

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()

        self.msg_sca_norms = nn.ModuleList()
        self.msg_vec_norms = nn.ModuleList()
        self.msg_vec2_norms = nn.ModuleList()
        self.fte_sca_norms = nn.ModuleList()
        self.fte_vec_norms = nn.ModuleList()
        self.fte_vec2_norms = nn.ModuleList()
        self.msg_dropouts = nn.ModuleList()
        self.fte_dropouts = nn.ModuleList()

        self.input_norm = nn.LayerNorm(input_dim)

        for i in range(num_layers):
            self.message_layers.append(
                EquiMessagePassing(
                    hidden_channels, num_radial, last_layer=(i == num_layers - 1), num_heads=num_heads
                )
            )
            self.FTEs.append(FTE(hidden_channels))

            self.msg_sca_norms.append(nn.LayerNorm(hidden_channels))
            self.fte_sca_norms.append(nn.LayerNorm(hidden_channels))

            self.msg_vec_norms.append(TensorLayerNorm(hidden_channels, trainable=trainable_vecnorm))
            self.msg_vec2_norms.append(TensorLayerNorm(hidden_channels, trainable=trainable_vecnorm))
            self.fte_vec_norms.append(TensorLayerNorm(hidden_channels, trainable=trainable_vecnorm))
            self.fte_vec2_norms.append(TensorLayerNorm(hidden_channels, trainable=trainable_vecnorm))

            self.msg_dropouts.append(nn.Dropout(dropout))
            self.fte_dropouts.append(nn.Dropout(dropout))


        self.edge_sca_proj = nn.Sequential(
            nn.Linear(3 * hidden_channels + num_radial, hidden_channels * 3),
            nn.SiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels),
        )
        self.mean_neighbor_pos = aggregate_pos(aggr='mean')
        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        nn.init.xavier_uniform_(self.radial_lin[0].weight)
        nn.init.xavier_uniform_(self.radial_lin[2].weight)
        nn.init.xavier_uniform_(self.edge_sca_proj[0].weight)
        self.edge_sca_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.edge_sca_proj[2].weight)
        self.edge_sca_proj[2].bias.data.fill_(0)

        self.neighbor_emb.reset_parameters()
        self.neighbor_embedding.reset_parameters()

    def forward(self, graph, input, all_loss=None, metric=None):


        pos = graph.pos
        if self.pos_require_grad:
            pos.requires_grad_()

        z_emb = self.z_emb(input)
        edge_index = graph.edge_index
        i, j = edge_index[0], edge_index[1]

        dist = torch.norm(pos[i] - pos[j], dim=-1)
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        soft_cutoff = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        radial_hidden = soft_cutoff.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(input, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 8, s.size(1), device=s.device)
        vec2 = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        edge_diff = _normalize(pos[i] - pos[j])
        edge_vec = self.tensor_init(edge_diff)

        mean_neighbor_pos = self.mean_neighbor_pos(pos, edge_index)
        node_diff = _normalize(pos - mean_neighbor_pos)
        node_cross = _normalize(torch.cross(pos, mean_neighbor_pos, dim=-1))
        node_vertical = torch.cross(node_diff, node_cross, dim=-1)
        node_frame = torch.cat((node_diff.unsqueeze(-1), node_cross.unsqueeze(-1), node_vertical.unsqueeze(-1)), dim=-1)

        A_i_j = torch.cat((s[i], s[j]), dim=-1) * soft_cutoff.unsqueeze(-1)
        edge_sca = self.edge_sca_proj(torch.cat((A_i_j, radial_hidden, radial_emb), dim=-1))

        for i in range(self.num_layers):

            ds, dvec, dve2, dedge_sca = self.message_layers[i](
                s, vec, vec2, edge_index, edge_sca, edge_diff, edge_vec, node_frame
            )

            dropout_layer = self.msg_dropouts[i]
            s = s + dropout_layer(ds)
            vec = vec + dvec
            vec2 = vec2 + dve2
            if dedge_sca is not None:
                edge_sca = edge_sca + dedge_sca


            ds, dvec, dvec2 = self.FTEs[i](s, vec, vec2)

            dropout_layer = self.fte_dropouts[i]
            s = s + dropout_layer(ds)
            vec = vec + dvec
            vec2 = vec2 + dve2

        node_feature = s + vec.sum() * 0 + vec2.sum() * 0

        return {
            "node_feature": node_feature
        }

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())



