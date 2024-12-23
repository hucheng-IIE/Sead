import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import math
# Graph Propagation models
from dgl.nn import GraphConv
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from utils import *
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# CompGCN based on direct graphs. We do not have inversed edges
class CompGCN_dg_AEPE(nn.Module):
    def __init__(self, node_in_feat, node_out_feat, rel_in_feat, rel_out_feat, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super().__init__()
        self.node_in_feat = node_in_feat
        self.node_out_feat = node_out_feat
        self.rel_in_feat = rel_in_feat
        self.rel_out_feat = rel_out_feat

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.W_mean = nn.Linear(self.node_in_feat,1)
        self.W_sum = nn.Linear(self.node_out_feat,1)

        if self.bias == True:
            self.bias_v = nn.Parameter(torch.Tensor(node_out_feat))
            # nn.init._xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.zeros_(self.bias_v)

        self.msg_inv_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias) # w@f(e_s,e_r) inverse
        if self.self_loop:
            self.msg_loop_linear = nn.Linear(node_in_feat, node_out_feat, bias=bias)     
        self.rel_linear = nn.Linear(rel_in_feat, rel_out_feat, bias=bias) # w@e_r
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, g, reverse=False, top_k=3): 
        
        def get_neighbor_counts(g):
            neighbor_counts = g.out_degrees().tolist() 
            for node, count in enumerate(neighbor_counts):
                print(f"Node {node} has {count} neighbors.")
        
        def custom_message_func(edges):
            message = edges.src['h'] * edges.data['text_h']
            return {'m': message}

        def apply_func(nodes):
            h = nodes.data['h'] * nodes.data['norm']
            if self.bias:
                h = h + self.bias_v
            if self.self_loop:
                h = self.msg_loop_linear(g.ndata['h'])
                # h = torch.mm(g.ndata['h'], self.loop_weight)
                if self.dropout is not None:
                    h = self.dropout(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        def apply_edge(edges):
            e_h = self.rel_linear(edges.data['text_h'])
            return {'text_h': e_h}

        def custom_agg_mean(nodes):
            neighbors_feature = nodes.mailbox['m'] # (n_nodes, n_neighbors, feature_dim)
            if neighbors_feature.shape[1] <=3:  #neighbor num
                feature = torch.mean(neighbors_feature, dim=1)
                return {'h_o_r': feature}
            else:
                neighbors_weights = F.softmax(neighbors_feature, dim=1)
                neighbors_weights_norm = self.W_mean(neighbors_weights)
                #select k neighors
                _, select_neighbors_indices = torch.topk(neighbors_weights_norm, k=top_k, dim=1)
                #select message
                select_neighbors_feature = torch.gather(neighbors_feature, dim=1, index=select_neighbors_indices.expand(-1, -1, neighbors_feature.shape[2]))
                #agg
                meg = torch.mean(select_neighbors_feature, dim=1)  # (n_nodes, feature_dim)
                return {'h_o_r': meg}

        def custom_agg_sum(nodes):
            neighbors_feature = nodes.mailbox['m'] # (n_nodes, n_neighbors, feature_dim)
            if neighbors_feature.shape[1] <=3:  #neighbor num
                feature = torch.sum(neighbors_feature, dim=1)
                return {'h': feature}
            else:
                neighbors_weights = F.softmax(neighbors_feature, dim=1)
                neighbors_weights_norm = self.W_sum(neighbors_weights)
                #select k neighors
                _, select_neighbors_indices = torch.topk(neighbors_weights_norm, k=top_k, dim=1)
                #select message
                select_neighbors_feature = torch.gather(neighbors_feature, dim=1, index=select_neighbors_indices.expand(-1, -1, neighbors_feature.shape[2]))
                #agg
                meg = torch.sum(select_neighbors_feature, dim=1)  # (n_nodes, feature_dim)
                return {'h': meg}

        #get_neighbor_counts(g)
        #g.update_all(fn.v_mul_e('h', 'e_h', 'm'), fn.mean('m', 'h_o_r'))
        g.update_all(custom_message_func, custom_agg_mean) 
        h_o_r = self.msg_inv_linear(g.ndata['h_o_r'])
        g.ndata['h_s_r_o'] = h_o_r
        #g.update_all(fn.copy_u(u='h_s_r_o', out='m'), fn.sum(msg='m', out='h'),apply_func)
        g.update_all(fn.copy_u(u='h_s_r_o', out='m'), custom_agg_sum, apply_func)
        g.apply_edges(apply_edge) 

# Graph Propagation models

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation:
            h = self.activation(h)
        return {'h' : h}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.0):
        super().__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, feature):
        def gcn_msg(edge):
            msg = edge.src['h'] * edge.data['w'].float()
            return {'m': msg}
 
        #feature = g.ndata['h']
        if self.dropout:
            feature = self.dropout(feature)

        g.ndata['h'] = feature
        g.update_all(gcn_msg, fn.sum(msg='m', out='h'))
        g.apply_nodes(func=self.apply_mod)
        return g.ndata['h']




