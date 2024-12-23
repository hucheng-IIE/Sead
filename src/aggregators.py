import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from propagations import *
from modules_f import *
import time
from itertools import groupby
from torch_scatter import scatter

from dgl.nn.pytorch.conv import GraphConv

# aggregator for event forecasting 
class CompGCN(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, sentence_size,  dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', device=None):
        super().__init__()
        self.h_dim = h_dim
        self.sentence_size = sentence_size
        self.text_embedding_size = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        self.device = device
        self.word_embeds = None
        self.global_emb = None
        self.ent_map = None
        self.rel_map = None
        self.word_graph_dict = None
        self.graph_dict = None
        self.sentence_embeddings_dict = None
        self.aggregator= aggregator_event_mtg(h_dim, dropout, num_ents, num_rels, self.sentence_size, self.text_embedding_size, seq_len, maxpool, attn, self.device)
        if use_gru:
            # self.encoder = nn.GRU(3*h_dim, h_dim, batch_first=True)
            self.encoder = nn.GRU(2*h_dim, h_dim, batch_first=True)
        else:
            # self.encoder = nn.RNN(3*h_dim, h_dim, batch_first=True)
            self.encoder = nn.RNN(4*h_dim, h_dim, batch_first=True)
        self.linear_r = nn.Linear(h_dim, self.num_rels)
        #self.linear_r = nn.Linear(h_dim*2, 1)
        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = nn.BCELoss()
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, ent_memory, rel_memory):
        pred, idx, _ = self.__get_pred_embeds(t_list, ent_memory, rel_memory)
        # loss = self.criterion(pred, true_prob_r[idx])
        # return loss
        return pred

    def __get_pred_embeds(self, t_list, ent_memory=None, rel_memory=None):
        # print(t_list)
        # sorted_t, idx = t_list.sort()
        # print(sorted_t)
        t_list = torch.tensor(t_list)
        embed_seq_tensor, len_non_zero = self.aggregator(t_list, ent_memory, rel_memory, self.ent_embeds,
                            self.rel_embeds, self.graph_dict, self.sentence_embeddings_dict)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)

        pred = self.linear_r(feature).squeeze(1)
        #pred = self.out_func(pred).squeeze(1)
        # print (pred)
        # quit()
        return pred, _, feature

    def predict(self, t_list, true_prob_r):
        pred, idx, feature = self.__get_pred_embeds(t_list)
        pred = pred.float()
        # print (pred.type())
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx].float())
        else:
            loss = None
        return loss, pred, feature
class aggregator_event(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, attn=''):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.se_aggr = GCN(768, h_dim, h_dim, 2, F.relu, dropout)

        # self.se_aggr = GCN(100, int(h_dim/2), h_dim, 2, F.relu, dropout)

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')       

    def forward(self, t_list, ent_embeds, rel_embeds, word_embeds, graph_dict, word_graph_dict, ent_map, rel_map):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]  # usually no duplicates

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)  
        # if torch.cuda.is_available():
        #     move_dgl_to_cuda(batched_g)
        # a = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]).data.cpu()
        # print(a.is_cuda)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)

        # word graph
        wg_list = [word_graph_dict[tim.item()] for tim in unique_t]
        batched_wg = dgl.batch(wg_list) 
        batched_wg = batched_wg.to(device)
        # if torch.cuda.is_available():
        #     move_dgl_to_cuda(batched_g)
        #     move_dgl_to_cuda(batched_wg)
        batched_wg.ndata['h'] = word_embeds[batched_wg.ndata['id']].view(-1, word_embeds.shape[1])

        self.re_aggr1(batched_g, False)
        self.re_aggr2(batched_g, False)

        batched_wg.ndata['h'] = self.se_aggr(batched_wg) 
        word_ids_wg = batched_wg.ndata['id'].view(-1).cpu().tolist()
        id_dict = dict(zip(word_ids_wg, list(range(len(word_ids_wg)))))
        
        # cpu operation for nodes
        g_node_embs = batched_g.ndata.pop('h').data.cpu()
        g_node_ids = batched_g.ndata['id'].view(-1)
        max_query_ent = 0
        num_nodes = len(g_node_ids)
        c_g_node_ids = g_node_ids.data.cpu().numpy()
        c_unique_ent_id = list(set(c_g_node_ids))
        ent_gidx_dict = {} # entid: [[gidx],[word_idx]]
        for ent_id in c_unique_ent_id:
            word_ids = ent_map[ent_id]
            word_idx = []
            for w in word_ids:
                try:
                    word_idx.append(id_dict[w])
                except:
                    continue
            if len(word_idx)>1:
                gidx = (c_g_node_ids==ent_id).nonzero()[0]
                word_idx = torch.LongTensor(word_idx)
                ent_gidx_dict[ent_id] = [gidx, word_idx]
                max_query_ent = max(max_query_ent, len(word_idx))

        # cpu operation on edges
        g_edge_embs = batched_g.edata.pop('e_h').data.cpu() ####
        g_edge_types = batched_g.edata['type'].view(-1)
        num_edges = len(g_edge_types)
        max_query_rel = 0
        c_g_edge_types = g_edge_types.data.cpu().numpy()
        c_unique_type_id = list(set(c_g_edge_types))
        type_gidx_dict = {}
        for type_id in c_unique_type_id:
            word_ids = rel_map[type_id]
            word_idx = []
            for w in word_ids:
                try:
                    word_idx.append(id_dict[w])
                except:
                    continue
            if len(word_idx)>1:
                gidx = (c_g_edge_types==type_id).nonzero()[0]
                word_idx = torch.LongTensor(word_idx)
                type_gidx_dict[type_id] = [gidx, word_idx]
                max_query_rel = max(max_query_rel, len(word_idx))

        max_query = max(max_query_ent, max_query_rel)
        # initialize a batch
        wg_node_embs = batched_wg.ndata['h'].data.cpu()
        Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
        Q_mx_rel = g_edge_embs.view(num_edges , 1, self.h_dim)
        Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
        H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))
        
        for ent in ent_gidx_dict:
            [gidx, word_idx] = ent_gidx_dict[ent]
            if len(gidx) > 1:
                embeds = wg_node_embs.index_select(0, word_idx)
                for i in gidx:
                    H_mx[i,range(len(word_idx)),:] = embeds
            else:
                H_mx[gidx,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
        
        for e_type in type_gidx_dict:
            [gidx, word_idx] = type_gidx_dict[e_type]
            if len(gidx) > 1:
                embeds = wg_node_embs.index_select(0, word_idx)
                for i in gidx:
                    H_mx[i,range(len(word_idx)),:] = embeds
            else:
                H_mx[gidx,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
        
        if torch.cuda.is_available():
            H_mx = H_mx.cuda()
            Q_mx = Q_mx.cuda()
        output, weights = self.attn(Q_mx, H_mx) # output (batch,1,h_dim)
        batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
        batched_g.edata['e_h'] = output[num_nodes:].view(-1, self.h_dim)
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
            global_word_info = dgl.max_nodes(batched_wg, 'h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')
            global_word_info = dgl.mean_nodes(batched_wg, 'h')
        global_node_info = torch.cat((global_node_info, global_edge_info, global_word_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 3*self.h_dim) 
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero

class aggregator_event_mtg(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, sentence_size, text_embedding_size,  seq_len=10, maxpool=1, attn='', device=None):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.device = device

        self.w_node = nn.Linear(h_dim*3,h_dim)
        self.w_rel = nn.Linear(h_dim*3,h_dim)

        self.sentence_size = sentence_size
        self.text_embedding_size = text_embedding_size

        self.textEmbeddingLayer = torch.nn.Linear(sentence_size, text_embedding_size)

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg_mtg(h_dim, out_feat, h_dim, out_feat, sentence_size, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg_mtg(out_feat, h_dim, out_feat, h_dim, sentence_size, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')

    def forward(self, t_list, ent_memory, rel_memory, ent_embeds, rel_embeds, graph_dict, sentence_embeddings_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]  # usually no duplicates
        for tim in t_list:
            length = times.index(tim)
            if (self.seq_len) <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)
        device = self.device
        #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
      
        batched_g.ndata['h'] = torch.cat((ent_embeds[batched_g.ndata['id']].squeeze(1),ent_memory[batched_g.ndata['id']].squeeze(1)), dim=1)
        batched_g.ndata['h'] = self.w_node(batched_g.ndata['h'])
        #batched_g.ndata['h']= ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1])
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']

        # story_ids = batched_g.edata['sid'].tolist()
        s_embeddings = batched_g.edata['text_emb']
        batched_g.edata['s_h'] = s_embeddings
        batched_g.edata['e_h'] = torch.cat([rel_embeds.index_select(0, type_data), rel_memory.index_select(0, type_data)], dim=1)
        batched_g.edata['e_h'] = self.w_rel(batched_g.edata['e_h'])
        #batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)
        self.re_aggr1(batched_g, False)
        self.re_aggr2(batched_g, False)

        # cpu operation for nodes
        g_node_embs = batched_g.ndata.pop('h').data.cpu()
        g_node_ids = batched_g.ndata['id'].view(-1)
        max_query_ent = 0
        num_nodes = len(g_node_ids)
        c_g_node_ids = g_node_ids.data.cpu().numpy()
        c_unique_ent_id = list(set(c_g_node_ids))
        ent_gidx_dict = {} # entid: [[gidx],[word_idx]]

        # cpu operation on edges
        g_edge_embs = batched_g.edata.pop('e_h').data.cpu() ####
        g_edge_types = batched_g.edata['type'].view(-1)
        num_edges = len(g_edge_types)
        max_query_rel = 0
        c_g_edge_types = g_edge_types.data.cpu().numpy()
        c_unique_type_id = list(set(c_g_edge_types))
        type_gidx_dict = {}

        # initialize a batch
        Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
        Q_mx_rel = g_edge_embs.view(num_edges , 1, self.h_dim)
        Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
        # H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))

        if torch.cuda.is_available():
            Q_mx = Q_mx.cuda()

        output = Q_mx
        batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
        batched_g.edata['e_h'] = output[num_nodes:].view(-1, self.h_dim)
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')

        global_node_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim)
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero

class DynamicGCN_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GCNLayer(in_feats, n_classes, activation, dropout))
        else:
            self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
            for i in range(n_layers - 2):
                self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
            self.layers.append(GCNLayer(n_hidden, n_classes, activation, dropout)) # activation or None

    def forward(self, g, features=None): # no reverse
        if features is None:
            h = g.ndata['h']
        else:
            h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GraphConv(in_feats, n_classes, activation=activation))
        else:
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
            for i in range(n_layers - 2):
                self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
            self.layers.append(GraphConv(n_hidden, n_classes, activation=activation)) # activation or None

    def forward(self, g, feature=False): # no reverse
        g = dgl.add_self_loop(g)
        def apply_func(nodes):
            h = nodes.data['h']
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        for layer in self.layers:
            g.ndata['h'] = layer(g, g.ndata['h'])
        g.apply_nodes(apply_func)

class glean_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers < 2:
            self.layers.append(GCNLayer(in_feats, n_classes, activation, dropout))
        else:
            self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
            for i in range(n_layers - 2):
                self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
            self.layers.append(GCNLayer(n_hidden, n_classes, activation, dropout)) # activation or None

    def forward(self, g, features=None): # no reverse
        if features is None:
            h = g.ndata['h']
        else:
            h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

class aggregator_event_CompGCN(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg_glean(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg_glean(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined   

    def forward(self, t_list, ent_embeds, rel_embeds, graph_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)

        self.re_aggr1(batched_g, False)
        self.re_aggr2(batched_g, False)
        
        # cpu operation for nodes
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')

        global_node_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim) 
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero
# aggregator for event forecasting 
class aggregator_event_glean(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, attn=''):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.w_text = nn.Linear(768,h_dim)
        self.se_aggr = glean_GCN(h_dim, h_dim, h_dim, 2, F.relu, dropout)

        # self.se_aggr = GCN(100, int(h_dim/2), h_dim, 2, F.relu, dropout)

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg_glean(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg_glean(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')       

    def forward(self, t_list, ent_embeds, rel_embeds, word_embeds, graph_dict, word_graph_dict, ent_map, rel_map):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        # nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        # t_list = t_list[nonzero_idx]  # usually no duplicates

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)
        # if torch.cuda.is_available():
        #     move_dgl_to_cuda(batched_g)
        # a = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]).data.cpu()
        # print(a.is_cuda)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)

        # word graph
        wg_list = [word_graph_dict[tim.item()] for tim in unique_t]
        batched_wg = dgl.batch(wg_list) 
        batched_wg = batched_wg.to(device)
        
        word_embeds = self.w_text(word_embeds.to(device))
        batched_wg.ndata['h'] = word_embeds[batched_wg.ndata['id']].view(-1, word_embeds.shape[1])

        self.re_aggr1(batched_g, False)
        self.re_aggr2(batched_g, False)

        batched_wg.ndata['h'] = self.se_aggr(batched_wg) 

        word_ids_wg = batched_wg.ndata['id'].view(-1).cpu().tolist()
        id_dict = dict(zip(word_ids_wg, list(range(len(word_ids_wg)))))
        
        # cpu operation for nodes
        g_node_embs = batched_g.ndata.pop('h').data.cpu()
        g_node_ids = batched_g.ndata['id'].view(-1)
        max_query_ent = 0
        num_nodes = len(g_node_ids)
        c_g_node_ids = g_node_ids.data.cpu().numpy()
        c_unique_ent_id = list(set(c_g_node_ids))
        ent_gidx_dict = {} # entid: [[gidx],[word_idx]]
        for ent_id in c_unique_ent_id:
            word_ids = ent_map[ent_id]
            word_idx = []
            for w in word_ids:
                try:
                    word_idx.append(id_dict[w])
                except:
                    continue
            if len(word_idx)>1:
                gidx = (c_g_node_ids==ent_id).nonzero()[0]
                word_idx = torch.LongTensor(word_idx)
                ent_gidx_dict[ent_id] = [gidx, word_idx]
                max_query_ent = max(max_query_ent, len(word_idx))

        # cpu operation on edges
        g_edge_embs = batched_g.edata.pop('e_h').data.cpu() ####
        g_edge_types = batched_g.edata['type'].view(-1)
        num_edges = len(g_edge_types)
        max_query_rel = 0
        c_g_edge_types = g_edge_types.data.cpu().numpy()
        c_unique_type_id = list(set(c_g_edge_types))
        type_gidx_dict = {}
        for type_id in c_unique_type_id:
            word_ids = rel_map[type_id]
            word_idx = []
            for w in word_ids:
                try:
                    word_idx.append(id_dict[w])
                except:
                    continue
            if len(word_idx)>1:
                gidx = (c_g_edge_types==type_id).nonzero()[0]
                word_idx = torch.LongTensor(word_idx)
                type_gidx_dict[type_id] = [gidx, word_idx]
                max_query_rel = max(max_query_rel, len(word_idx))

        max_query = max(max_query_ent, max_query_rel)
        # initialize a batch
        wg_node_embs = batched_wg.ndata['h'].data.cpu()
        Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
        Q_mx_rel = g_edge_embs.view(num_edges , 1, self.h_dim)
        Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
        H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))
        
        for ent in ent_gidx_dict:
            [gidx, word_idx] = ent_gidx_dict[ent]
            if len(gidx) > 1:
                embeds = wg_node_embs.index_select(0, word_idx)
                for i in gidx:
                    H_mx[i,range(len(word_idx)),:] = embeds
            else:
                H_mx[gidx,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
        
        for e_type in type_gidx_dict:
            [gidx, word_idx] = type_gidx_dict[e_type]
            if len(gidx) > 1:
                embeds = wg_node_embs.index_select(0, word_idx)
                for i in gidx:
                    H_mx[i,range(len(word_idx)),:] = embeds
            else:
                H_mx[gidx,range(len(word_idx)),:] = wg_node_embs.index_select(0, word_idx)
        
        if torch.cuda.is_available():
            H_mx = H_mx.cuda()
            Q_mx = Q_mx.cuda()
        output, weights = self.attn(Q_mx, H_mx) # output (batch,1,h_dim)
        batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
        batched_g.edata['e_h'] = output[num_nodes:].view(-1, self.h_dim)
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
            global_word_info = dgl.max_nodes(batched_wg, 'h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')
            global_word_info = dgl.mean_nodes(batched_wg, 'h')
        global_node_info = torch.cat((global_node_info, global_edge_info, global_word_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 3*self.h_dim) 
        # global_node_info = torch.cat((global_node_info, global_edge_info), -1)
        # embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim) 
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero

class aggregator_event_PECF(nn.Module):
    def __init__(self, node_in_feat, dropout, num_nodes, num_rels, seq_len=7, maxpool=1,n_layers=2,rnn_layers=1,node_embeds=None,rel_embeds=None,text_emb_dim=None):
        super().__init__()
        self.node_in_feat = node_in_feat  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.rnn_layers = rnn_layers
        self.node_embeds = node_embeds
        self.rel_embeds = rel_embeds
        self.text_w = nn.Linear(text_emb_dim,node_in_feat,bias=True)
        self.activate = nn.ReLU()
        self.edge_w = nn.Linear(node_in_feat*3,node_in_feat,bias=True)

        self.encoder = nn.GRU(node_in_feat, node_in_feat, batch_first=True)

        out_feat = int(node_in_feat // 2)
        hid_dim = int(node_in_feat *2)

        self.re_aggr1 = RGCN_dg(node_in_feat, hid_dim,out_feat,node_in_feat,out_feat,n_layers,num_rels,regularizer="basis",num_bases=None,
                                use_bias=True,activation=F.relu, use_self_loop=True,
                                layer_norm=False,low_mem=False, dropout=0.2, text_emb_dim=text_emb_dim)
        self.re_aggr2 = RGCN_dg(out_feat, hid_dim,node_in_feat,out_feat,node_in_feat,n_layers,num_rels,regularizer="basis",num_bases=None,
                                use_bias=True,activation=F.relu, use_self_loop=True,
                                layer_norm=False,low_mem=False, dropout=0.2, text_emb_dim=text_emb_dim)
        # self.re_aggr1 = CompGCN_dg_PECF(node_in_feat, hid_dim, node_in_feat, hid_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        # self.re_aggr2 = CompGCN_dg_PECF(hid_dim, node_in_feat, hid_dim, node_in_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        
        # self.re_aggr1 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        # self.re_aggr2 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        
    def __get_rel_embedding(self,batch_g):
        init_rel_embeds = torch.zeros(self.num_rels, self.node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(batch_g.edata['e_h'],batch_g_rel,dim=0,reduce="mean")  # shape=(max rel in batch_G, relaion text emb dim)
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]
        return init_rel_embeds

    def forward(self, t_list, graph_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)
        
        #init embeds
        init_node_embeds = self.node_embeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #each time
        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        #entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(device)   #gpu
        
        #init node embedding
        node_id = batched_graph.ndata['id']
        batched_graph.ndata['h'] = init_node_embeds[node_id].view(-1, init_node_embeds.shape[1]).to(device)
        batched_graph.edata['text_h'] = self.text_w(batched_graph.edata['text_emb'])

        #caclulate node embedding
        self.re_aggr1(batched_graph)
        self.re_aggr2(batched_graph)
        #caclulate edge embedding
        batch_g_src, batch_g_dst = batched_graph.edges()
        batch_g_src_nid = batch_g_src.long()
        batch_g_dst_nid = batch_g_dst.long()

        batch_dyn_node2src_embeds = batched_graph.ndata['h'][batch_g_src_nid].to(device)
        batch_dyn_node2dst_embeds = batched_graph.ndata['h'][batch_g_dst_nid].to(device)
        batch_g_edge_text_embeds = batched_graph.edata['text_h']

        batched_graph.edata['e_h'] = torch.cat([batch_dyn_node2src_embeds,batch_g_edge_text_embeds,
                                                        batch_dyn_node2dst_embeds], dim=-1)
        batched_graph.edata['e_h'] = self.edge_w(batched_graph.edata['e_h'])
        batched_graph.edata['e_h'] = self.activate(batched_graph.edata['e_h'])

        #print("batched_graph.edata['e_h']",batched_graph.edata['e_h'])
        g_list = dgl.unbatch(batched_graph)
        batched_embed_seq_tensor = []
        embed_seq_tensor = []
        for i, times in enumerate(time_list):
            batched_embed_seq_tensor = []
            for j, t in enumerate(times):
               batched_embed_seq_tensor.append(self.__get_rel_embedding(g_list[time_to_idx[t.item()]]))
            batched_embed_seq_tensor = torch.stack(batched_embed_seq_tensor).to(device)
            #(num_rels,seq_len,feature)
            output,hn = self.encoder(batched_embed_seq_tensor.transpose(0, 1))
            #(num_rels,feature)
            rel_embeds = hn.squeeze(0)
            embed_seq_tensor.append(rel_embeds)
        embed_seq_tensor = torch.stack(embed_seq_tensor).to(device)
        

        return embed_seq_tensor
    
class aggregator_event_DynamicGCN(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, vocab_size, seq_len=10, maxpool=1, output=1):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.output = output
        self.w_text = nn.Linear(768,h_dim)
        self.se_aggr = DynamicGCN_GCN(h_dim, h_dim, h_dim, 2, F.relu, dropout)
        self.bn = nn.BatchNorm1d(h_dim)
        self.temporal_cell = TemporalEncoding(h_dim)

        self.Lasr_layer_se_aggr = DynamicGCN_GCN(h_dim, h_dim, self.output, 2, F.relu, dropout)
        self.Lasr_layer_bn = nn.BatchNorm1d(self.output) 
        self.mask = MaskLinear(vocab_size, self.output)

    def forward(self, t_list, word_embeds, word_graph_dict):
        times = list(word_graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []
        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # word graph
        wg_list = [word_graph_dict[tim.item()] for tim in unique_t]
        #init last_x
        word_embeds = self.w_text(word_embeds)
        last_x = word_embeds
        for i in range(len(wg_list)):
            batch_g = wg_list[i].to(device)
            node_id = batch_g.ndata['id'].squeeze(1)
            x = last_x[node_id].to(device)
            if i < len(wg_list)-1:
                x = self.se_aggr(batch_g,x)
                x = self.bn(x)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.temporal_cell(x,word_embeds[node_id])
                last_x[node_id] = x
            else:
                x = self.Lasr_layer_se_aggr(batch_g,x)
                x = self.Lasr_layer_bn(x)
                x = F.relu(x)
                x = self.mask(x, node_id)
                x = torch.sigmoid(x) 
                return x

class aggregator_event_TGCN(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, vocab_size, seq_len=10, maxpool=1, output=1):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.output = output
        self.w_text = nn.Linear(768,h_dim)
        self.se_aggr = glean_GCN(h_dim, h_dim, h_dim, 2, F.relu, dropout)
        self.bn = nn.BatchNorm1d(h_dim)
        #self.temporal_cell = TemporalEncoding(h_dim)
        self.temporal_cell = nn.GRU(h_dim,h_dim,batch_first=True)

        self.Lasr_layer_se_aggr = glean_GCN(h_dim, h_dim, self.output, 2, F.relu, dropout)
        self.Lasr_layer_bn = nn.BatchNorm1d(self.output) 
        self.mask = MaskLinear(vocab_size, self.output)
        # self.Lasr_layer_bn = nn.BatchNorm1d(self.output) 
        # self.mask = MaskLinear(vocab_size, self.output)

    def forward(self, t_list, word_embeds, word_graph_dict):
        times = list(word_graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []
        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # word graph get 
        wg_list = [word_graph_dict[tim.item()] for tim in unique_t]
        word_embeds = self.w_text(word_embeds)
        last_x = word_embeds
        for i in range(len(wg_list)):
            batch_g = wg_list[i].to(device)
            node_id = batch_g.ndata['id'].squeeze(1)
            x = last_x[node_id].to(device)
            if i < len(wg_list)-1:
                x = self.se_aggr(batch_g,x)
                x = self.bn(x)
                x = F.relu(x)
                x = self.dropout(x)
                # x = self.temporal_cell(x,word_embeds[node_id])
                gru_input = x.unsqueeze(1).to(device)
                h0 = word_embeds[node_id].unsqueeze(1).to(device)
                output, hn = self.temporal_cell(gru_input, h0.transpose(0, 1).contiguous())#(transpose to get (num_layers,batch_size,feat_dim))
                x = hn.transpose(0, 1).squeeze()
                last_x[node_id] = x
            else:
                x = self.Lasr_layer_se_aggr(batch_g,x)
                x = self.Lasr_layer_bn(x)
                x = F.relu(x)
                x = self.mask(x, node_id)
                x = torch.sigmoid(x) 
                return x
        return x

class aggregator_event_tRGCN(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1, n_layers=2):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)

        self.re_aggr1 = tRGCN_dg(h_dim, out_feat,out_feat,h_dim,out_feat,n_layers,num_rels,regularizer="basis",num_bases=None,
                                use_bias=True,activation=F.relu, use_self_loop=True,
                                layer_norm=False,low_mem=False, dropout=0.2)
        self.re_aggr2 = tRGCN_dg(out_feat, h_dim,h_dim,out_feat,h_dim,n_layers,num_rels,regularizer="basis",num_bases=None,
                                use_bias=True,activation=F.relu, use_self_loop=True,
                                layer_norm=False,low_mem=False, dropout=0.2)
        
        # self.re_aggr1 = CompGCN_dg_glean(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        # self.re_aggr2 = CompGCN_dg_glean(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined   

    def forward(self, t_list, ent_embeds, rel_embeds, graph_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)

        self.re_aggr1(batched_g)
        self.re_aggr2(batched_g)
        
        # cpu operation for nodes
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')

        global_node_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim) 
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero

class aggregator_event_GCN(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, seq_len=10, maxpool=1):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool

        if maxpool == 1:
            self.dgl_global_edge_f = dgl.max_edges
            self.dgl_global_node_f = dgl.max_nodes
        else:
            self.dgl_global_edge_f = dgl.mean_edges
            self.dgl_global_node_f = dgl.mean_nodes

        out_feat = int(h_dim // 2)
        # self.re_aggr1 = CompGCN_dg_glean(h_dim, out_feat, h_dim, out_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        # self.re_aggr2 = CompGCN_dg_glean(out_feat, h_dim, out_feat, h_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr1 = GCN(h_dim, h_dim, out_feat, 2, F.relu, dropout)
        self.re_aggr2 = GCN(out_feat, h_dim, h_dim, 2, F.relu, dropout)

    def forward(self, t_list, ent_embeds, rel_embeds, graph_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h'] = ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]) 
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']
        batched_g.edata['e_h'] = rel_embeds.index_select(0, type_data)

        self.re_aggr1(batched_g, False)
        #elf.re_aggr2(batched_g, False)
        
        # cpu operation for nodes
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')

        global_node_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim) 
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero

# aggregator for event forecasting 
class aggregator_event_AEPE(nn.Module):
    def __init__(self, node_in_feat, dropout, num_nodes, num_rels, seq_len=7, maxpool=1,n_layers=2,node_embeds=None,rel_embeds=None,text_emb_dim=None):
        super().__init__()
        self.node_in_feat = node_in_feat  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.node_embeds = node_embeds
        self.rel_embeds = rel_embeds
        self.text_w = nn.Linear(text_emb_dim,node_in_feat,bias=True)
        self.activate = nn.ReLU()
        self.edge_w = nn.Linear(node_in_feat*3,node_in_feat,bias=True)

        self.encoder = nn.GRU(node_in_feat, node_in_feat, batch_first=True)
        self.n_layers = n_layers

        out_feat = int(node_in_feat // 2)
        hid_dim = int(node_in_feat *2)

        self.re_aggr1 = CompGCN_dg_AEPE(node_in_feat, hid_dim, node_in_feat, hid_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg_AEPE(hid_dim, node_in_feat, hid_dim, node_in_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        
        # self.re_aggr1 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        # self.re_aggr2 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        
    def __get_rel_embedding(self,batch_g):
        init_rel_embeds = torch.zeros(self.num_rels, self.node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(batch_g.edata['e_h'],batch_g_rel,dim=0,reduce="mean")  # shape=(max rel in batch_G, relaion text emb dim)
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]
        return init_rel_embeds

    def forward(self, t_list, graph_dict, add_graph_noise=False):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len+1:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)
        
        #init embeds
        init_node_embeds = self.node_embeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #each time
        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        #entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        #add noise to graph
        if add_graph_noise:
            g_list = add_noise_to_graphs_graphcl(g_list)
        
        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(device)   #gpu
        
        #init node embedding
        node_id = batched_graph.ndata['id']
        batched_graph.ndata['h'] = init_node_embeds[node_id].view(-1, init_node_embeds.shape[1]).to(device)
        batched_graph.edata['text_h'] = self.text_w(batched_graph.edata['text_emb'])

        #update node embedding
        for _ in range(self.n_layers):
            self.re_aggr1(batched_graph)
            self.re_aggr2(batched_graph)
        #caclulate edge embedding
        batch_g_src, batch_g_dst = batched_graph.edges()
        batch_g_src_nid = batch_g_src.long()
        batch_g_dst_nid = batch_g_dst.long()

        batch_dyn_node2src_embeds = batched_graph.ndata['h'][batch_g_src_nid].to(device)
        batch_dyn_node2dst_embeds = batched_graph.ndata['h'][batch_g_dst_nid].to(device)
        batch_g_edge_text_embeds = batched_graph.edata['text_h']

        batched_graph.edata['e_h'] = torch.cat([batch_dyn_node2src_embeds,batch_g_edge_text_embeds,
                                                        batch_dyn_node2dst_embeds], dim=-1)
        batched_graph.edata['e_h'] = self.edge_w(batched_graph.edata['e_h'])
        batched_graph.edata['e_h'] = self.activate(batched_graph.edata['e_h'])

        #print("batched_graph.edata['e_h']",batched_graph.edata['e_h'])
        g_list = dgl.unbatch(batched_graph)
        batched_embed_seq_tensor = []
        embed_seq_tensor = []
        for i, times in enumerate(time_list):
            batched_embed_seq_tensor = []
            for j, t in enumerate(times):
               batched_embed_seq_tensor.append(self.__get_rel_embedding(g_list[time_to_idx[t.item()]]))
            batched_embed_seq_tensor = torch.stack(batched_embed_seq_tensor).to(device)
            #(num_rels,seq_len,feature)
            output,hn = self.encoder(batched_embed_seq_tensor.transpose(0, 1))
            #(num_rels,feature)
            rel_embeds = hn.squeeze(0)
            embed_seq_tensor.append(rel_embeds)
        embed_seq_tensor = torch.stack(embed_seq_tensor).to(device)
        
        return embed_seq_tensor

class aggregator_event_GACN(nn.Module):
    def __init__(self, node_in_feat, dropout, num_nodes, num_rels, seq_len=7, maxpool=1,n_layers=2,node_embeds=None,rel_embeds=None,text_emb_dim=None):
        super().__init__()
        self.node_in_feat = node_in_feat  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.node_embeds = node_embeds
        self.rel_embeds = rel_embeds
        self.text_w = nn.Linear(text_emb_dim,node_in_feat,bias=True)
        self.activate = nn.ReLU()
        self.edge_w = nn.Linear(node_in_feat*3,node_in_feat,bias=True)

        self.encoder = nn.GRU(node_in_feat, node_in_feat, batch_first=True)
        self.n_layers = n_layers

        out_feat = int(node_in_feat // 2)
        hid_dim = int(node_in_feat *2)

        # self.re_aggr1 = CompGCN_dg_AEPE(node_in_feat, hid_dim, node_in_feat, hid_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        # self.re_aggr2 = CompGCN_dg_AEPE(hid_dim, node_in_feat, hid_dim, node_in_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        
        self.re_aggr1 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        self.re_aggr2 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        
    def __get_rel_embedding(self,batch_g):
        init_rel_embeds = torch.zeros(self.num_rels, self.node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(batch_g.edata['e_h'],batch_g_rel,dim=0,reduce="mean")  # shape=(max rel in batch_G, relaion text emb dim)
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]
        return init_rel_embeds

    def forward(self, t_list, graph_dict, add_graph_noise=False):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)
        
        #init embeds
        init_node_embeds = self.node_embeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #each time
        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        #entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        #add noise to graph
        if add_graph_noise:
            g_list = add_noise_to_graphs(g_list)
        
        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(device)   #gpu
        
        #init node embedding
        node_id = batched_graph.ndata['id']
        batched_graph.ndata['h'] = init_node_embeds[node_id].view(-1, init_node_embeds.shape[1]).to(device)
        batched_graph.edata['text_h'] = self.text_w(batched_graph.edata['text_emb'])

        #update node embedding
        for _ in range(self.n_layers):
            self.re_aggr1(batched_graph)
            self.re_aggr2(batched_graph)
        #caclulate edge embedding
        batch_g_src, batch_g_dst = batched_graph.edges()
        batch_g_src_nid = batch_g_src.long()
        batch_g_dst_nid = batch_g_dst.long()

        batch_dyn_node2src_embeds = batched_graph.ndata['h'][batch_g_src_nid].to(device)
        batch_dyn_node2dst_embeds = batched_graph.ndata['h'][batch_g_dst_nid].to(device)
        batch_g_edge_text_embeds = batched_graph.edata['text_h']

        batched_graph.edata['e_h'] = torch.cat([batch_dyn_node2src_embeds,batch_g_edge_text_embeds,
                                                        batch_dyn_node2dst_embeds], dim=-1)
        batched_graph.edata['e_h'] = self.edge_w(batched_graph.edata['e_h'])
        batched_graph.edata['e_h'] = self.activate(batched_graph.edata['e_h'])

        #print("batched_graph.edata['e_h']",batched_graph.edata['e_h'])
        g_list = dgl.unbatch(batched_graph)
        batched_embed_seq_tensor = []
        embed_seq_tensor = []
        for i, times in enumerate(time_list):
            batched_embed_seq_tensor = []
            for j, t in enumerate(times):
               batched_embed_seq_tensor.append(self.__get_rel_embedding(g_list[time_to_idx[t.item()]]))
            batched_embed_seq_tensor = torch.stack(batched_embed_seq_tensor).to(device)
            #(num_rels,seq_len,feature)
            output,hn = self.encoder(batched_embed_seq_tensor.transpose(0, 1))
            #(num_rels,feature)
            rel_embeds = hn.squeeze(0)
            embed_seq_tensor.append(rel_embeds)
        embed_seq_tensor = torch.stack(embed_seq_tensor).to(device)
        
        return embed_seq_tensor

class aggregator_event_GraphCL(nn.Module):
    def __init__(self, node_in_feat, dropout, num_nodes, num_rels, seq_len=7, maxpool=1,n_layers=2,node_embeds=None,rel_embeds=None,text_emb_dim=None):
        super().__init__()
        self.node_in_feat = node_in_feat  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.node_embeds = node_embeds
        self.rel_embeds = rel_embeds
        self.text_w = nn.Linear(text_emb_dim,node_in_feat,bias=True)
        self.activate = nn.ReLU()
        self.edge_w = nn.Linear(node_in_feat*3,node_in_feat,bias=True)

        self.encoder = nn.GRU(node_in_feat, node_in_feat, batch_first=True)
        self.n_layers = n_layers

        out_feat = int(node_in_feat // 2)
        hid_dim = int(node_in_feat *2)

        # self.re_aggr1 = CompGCN_dg_AEPE(node_in_feat, hid_dim, node_in_feat, hid_dim, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        # self.re_aggr2 = CompGCN_dg_AEPE(hid_dim, node_in_feat, hid_dim, node_in_feat, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        
        self.re_aggr1 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        self.re_aggr2 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        
    def __get_rel_embedding(self,batch_g):
        init_rel_embeds = torch.zeros(self.num_rels, self.node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(batch_g.edata['e_h'],batch_g_rel,dim=0,reduce="mean")  # shape=(max rel in batch_G, relaion text emb dim)
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]
        return init_rel_embeds

    def forward(self, t_list, graph_dict, add_graph_noise=False):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)
        
        #init embeds
        init_node_embeds = self.node_embeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #each time
        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        #entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        #add noise to graph
        if add_graph_noise:
            g_list = add_noise_to_graphs_graphcl(g_list)
        
        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(device)   #gpu
        
        #init node embedding
        node_id = batched_graph.ndata['id']
        batched_graph.ndata['h'] = init_node_embeds[node_id].view(-1, init_node_embeds.shape[1]).to(device)
        batched_graph.edata['text_h'] = self.text_w(batched_graph.edata['text_emb'])

        #update node embedding
        for _ in range(self.n_layers):
            self.re_aggr1(batched_graph)
            self.re_aggr2(batched_graph)
        #caclulate edge embedding
        batch_g_src, batch_g_dst = batched_graph.edges()
        batch_g_src_nid = batch_g_src.long()
        batch_g_dst_nid = batch_g_dst.long()

        batch_dyn_node2src_embeds = batched_graph.ndata['h'][batch_g_src_nid].to(device)
        batch_dyn_node2dst_embeds = batched_graph.ndata['h'][batch_g_dst_nid].to(device)
        batch_g_edge_text_embeds = batched_graph.edata['text_h']

        batched_graph.edata['e_h'] = torch.cat([batch_dyn_node2src_embeds,batch_g_edge_text_embeds,
                                                        batch_dyn_node2dst_embeds], dim=-1)
        batched_graph.edata['e_h'] = self.edge_w(batched_graph.edata['e_h'])
        batched_graph.edata['e_h'] = self.activate(batched_graph.edata['e_h'])

        #print("batched_graph.edata['e_h']",batched_graph.edata['e_h'])
        g_list = dgl.unbatch(batched_graph)
        batched_embed_seq_tensor = []
        embed_seq_tensor = []
        for i, times in enumerate(time_list):
            batched_embed_seq_tensor = []
            for j, t in enumerate(times):
               batched_embed_seq_tensor.append(self.__get_rel_embedding(g_list[time_to_idx[t.item()]]))
            batched_embed_seq_tensor = torch.stack(batched_embed_seq_tensor).to(device)
            #(num_rels,seq_len,feature)
            output,hn = self.encoder(batched_embed_seq_tensor.transpose(0, 1))
            #(num_rels,feature)
            rel_embeds = hn.squeeze(0)
            embed_seq_tensor.append(rel_embeds)
        embed_seq_tensor = torch.stack(embed_seq_tensor).to(device)
        
        return embed_seq_tensor

class aggregator_event_GCA(nn.Module):
    def __init__(self, node_in_feat, dropout, num_nodes, num_rels, seq_len=7, maxpool=1,n_layers=2,node_embeds=None,rel_embeds=None,text_emb_dim=None):
        super().__init__()
        self.node_in_feat = node_in_feat  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.node_embeds = node_embeds
        self.rel_embeds = rel_embeds
        self.text_w = nn.Linear(text_emb_dim,node_in_feat,bias=True)
        self.activate = nn.ReLU()
        self.edge_w = nn.Linear(node_in_feat*3,node_in_feat,bias=True)

        self.encoder = nn.GRU(node_in_feat, node_in_feat, batch_first=True)
        self.n_layers = n_layers

        out_feat = int(node_in_feat // 2)
        hid_dim = int(node_in_feat *2)
        
        self.re_aggr1 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        self.re_aggr2 = GCN_PECF(node_in_feat, hid_dim, node_in_feat, node_in_feat, node_in_feat, n_layers, activation=F.relu) # to be defined
        
    def __get_rel_embedding(self,batch_g):
        init_rel_embeds = torch.zeros(self.num_rels, self.node_in_feat).cuda()
        #caclulate rel embedding
        batch_g_rel = batch_g.edata['rel_type'].long()
        batch_g_uniq_rel = torch.unique(batch_g_rel, sorted=True)
        # aggregate edge embedding by relation
        batch_edge_emb_avg_by_rel_ = scatter(batch_g.edata['e_h'],batch_g_rel,dim=0,reduce="mean")  # shape=(max rel in batch_G, relaion text emb dim)
        #get uniq rel embedding
        init_rel_embeds[batch_g_uniq_rel] = batch_edge_emb_avg_by_rel_[batch_g_uniq_rel]
        return init_rel_embeds

    def forward(self, t_list, graph_dict, add_graph_noise=False):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_list = []
        len_non_zero = []

        for tim in t_list:
            length = times.index(tim)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)
        
        #init embeds
        init_node_embeds = self.node_embeds
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #each time
        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        #entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        #add noise to graph
        if add_graph_noise:
            g_list = add_noise_to_graphs_gca(g_list,init_node_embeds)
        
        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(device)   #gpu
        
        #init node embedding
        node_id = batched_graph.ndata['id']
        batched_graph.ndata['h'] = init_node_embeds[node_id].view(-1, init_node_embeds.shape[1]).to(device)
        batched_graph.edata['text_h'] = self.text_w(batched_graph.edata['text_emb'])

        #update node embedding
        for _ in range(self.n_layers):
            self.re_aggr1(batched_graph)
            self.re_aggr2(batched_graph)
        #caclulate edge embedding
        batch_g_src, batch_g_dst = batched_graph.edges()
        batch_g_src_nid = batch_g_src.long()
        batch_g_dst_nid = batch_g_dst.long()

        batch_dyn_node2src_embeds = batched_graph.ndata['h'][batch_g_src_nid].to(device)
        batch_dyn_node2dst_embeds = batched_graph.ndata['h'][batch_g_dst_nid].to(device)
        batch_g_edge_text_embeds = batched_graph.edata['text_h']

        batched_graph.edata['e_h'] = torch.cat([batch_dyn_node2src_embeds,batch_g_edge_text_embeds,
                                                        batch_dyn_node2dst_embeds], dim=-1)
        batched_graph.edata['e_h'] = self.edge_w(batched_graph.edata['e_h'])
        batched_graph.edata['e_h'] = self.activate(batched_graph.edata['e_h'])

        #print("batched_graph.edata['e_h']",batched_graph.edata['e_h'])
        g_list = dgl.unbatch(batched_graph)
        batched_embed_seq_tensor = []
        embed_seq_tensor = []
        for i, times in enumerate(time_list):
            batched_embed_seq_tensor = []
            for j, t in enumerate(times):
               batched_embed_seq_tensor.append(self.__get_rel_embedding(g_list[time_to_idx[t.item()]]))
            batched_embed_seq_tensor = torch.stack(batched_embed_seq_tensor).to(device)
            #(num_rels,seq_len,feature)
            output,hn = self.encoder(batched_embed_seq_tensor.transpose(0, 1))
            #(num_rels,feature)
            rel_embeds = hn.squeeze(0)
            embed_seq_tensor.append(rel_embeds)
        embed_seq_tensor = torch.stack(embed_seq_tensor).to(device)
        
        return embed_seq_tensor