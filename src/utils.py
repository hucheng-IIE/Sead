import numpy as np
import os
from math import log
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

import dgl
import torch
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, zero_one_loss,balanced_accuracy_score,accuracy_score
from sklearn.metrics import jaccard_score
from event_data_processing import *

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)
  
class MergeLayer_tgn(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    #x = torch.cat([x1, x2], dim=1)
    x = self.fc1(x)
    h = self.act(self.fc2(x))
    return h


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

def get_sentence_embeddings(graph_dict, source_cache, edge_times, device):
    # n = len(story_ids)
    # sentence_embeddings = torch.nn.Parameter(torch.zeros(n, embedding_size), requires_grad=True).to(device)
    # nn.init.xavier_uniform_(sentence_embeddings, gain=nn.init.calculate_gain('relu'))
    n = len(source_cache)
    edge_times = edge_times.tolist()
    time = list(set(edge_times))
    g_list = graph_dict[time[0]]
    sentence_embeddings = g_list.edata['text_emb'].to(device)
    if len(sentence_embeddings) != n:
       print("!!")
       print(time)
       sentence_embeddings = torch.nn.Parameter(torch.zeros(n, 768), requires_grad=True).to(device)
       nn.init.xavier_uniform_(sentence_embeddings, gain=nn.init.calculate_gain('relu'))
    return sentence_embeddings

def node_norm_to_edge_norm(G):
    G = G.local_var()
    G.apply_edges(lambda edges: {'norm': edges.dst['norm']})

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_y_data(data, target_rels):
    """ (# of s-related triples) / (total # of triples) """

    time_l = list(set(data[:,-1]))
    time_l = sorted(time_l,reverse=False)
    y_data = []
    for cur_t in time_l:
        has_Y = 0
        triples = get_data_with_t(data, cur_t)
        r_arr = triples[:,1]
        for rel in r_arr:
            if rel in target_rels:
                has_Y = 1
                break
        y_data.append(has_Y)
    # y_data = torch.Tensor(y_data)
    return y_data

def get_data_with_t(data, time):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == time]
    return np.array(triples)


def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None

 
def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    #get times
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)
 
'''
Customized collate function for Pytorch data loader
'''
def collate_2(batch):
    batch_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    return [batch_data, y_data]
 
def collate_4(batch):
    batch_data = [item[0] for item in batch]
    s_prob = [item[1] for item in batch]
    r_prob = [item[2] for item in batch]
    o_prob = [item[3] for item in batch]
    return [batch_data, s_prob, r_prob, o_prob]

def collate_6(batch):
    inp0 = [item[0] for item in batch]
    inp1 = [item[1] for item in batch]
    inp2 = [item[2] for item in batch]
    inp3 = [item[3] for item in batch]
    inp4 = [item[4] for item in batch]
    inp5 = [item[5] for item in batch]
    return [inp0, inp1, inp2, inp3, inp4, inp5]


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    if torch.cuda.is_available():
        g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
        g.edata.update({k: cuda(g.edata[k]) for k in g.edata})

 
'''
Get sorted r to make batch for RNN (sorted by length)
'''
def get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False):
    r_hist_len = torch.LongTensor(list(map(len, r_hist)))
    if torch.cuda.is_available():
        r_hist_len = r_hist_len.cuda()
    r_len, idx = r_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(r_len,as_tuple=False))
    r_len_non_zero = r_len[:num_non_zero]
    idx_non_zero = idx[:num_non_zero]  
    idx_zero = idx[num_non_zero-1:]  
    if torch.max(r_hist_len) == 0:
        return None, None, r_len_non_zero, [], idx, num_non_zero
    r_sorted = r[idx]
    r_hist_t_sorted = [r_hist_t[i] for i in idx]
    g_list = []
    wg_list = []
    r_ids_graph = []
    r_ids = 0 # first edge is r 
    for t_i in range(len(r_hist_t_sorted[:num_non_zero])):
        for tim in r_hist_t_sorted[t_i]:
            try:
                wg_list.append(word_graph_dict[r_sorted[t_i].item()][tim])
            except:
                pass

            try:
                sub_g = graph_dict[r_sorted[t_i].item()][tim]
                if sub_g is not None:
                    g_list.append(sub_g)
                    r_ids_graph.append(r_ids) 
                    r_ids += sub_g.number_of_edges()
            except:
                continue
    if len(wg_list) > 0:
        batched_wg = dgl.batch(wg_list)
    else:
        batched_wg = None
    if len(g_list) > 0:
        batched_g = dgl.batch(g_list)
    else:
        batched_g = None
    
    return batched_g, batched_wg, r_len_non_zero, r_ids_graph, idx, num_non_zero
 
def get_neighbor_finder(inPath, fileName, fileName2, fileName3, uniform, num_node):
    adj_list = [[] for _ in range(num_node)]
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            # adj_list[destination].append((source, edge_idx, timestamp))
    with open(os.path.join(inPath, fileName2), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            # adj_list[destination].append((source, edge_idx, timestamp))
    with open(os.path.join(inPath, fileName3), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])-1
            time = int(line_split[3])
            adj_list[head].append((tail, rel, time))
            # adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)

class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst


'''
Loss function
'''
# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1) # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

 
'''
Generate/get (r,t,s_count, o_count) datasets 
'''
def get_scaled_tr_dataset(num_nodes, path='../data/', dataset='india', set_name='train', seq_len=7, num_r=None):
    import pandas as pd
    from scipy import sparse
    file_path = '{}{}/tr_data_{}_sl{}_rand_{}.npy'.format(path, dataset, set_name, seq_len, num_r)
    if not os.path.exists(file_path):
        print(file_path,'not exists STOP for now')
        exit()
    else:
        print('load tr_data ...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o
 
'''
Empirical distribution
'''
def get_true_distributions(path, data, num_nodes, num_rels, dataset='india', set_name='train'):
    """ (# of s-related triples) / (total # of triples) """
     
    file_path = '{}{}/true_probs_{}.npy'.format(path, dataset, set_name)
    if not os.path.exists(file_path):
        print('build true distributions...',dataset,set_name)
        time_l = list(set(data[:,-1]))
        time_l = sorted(time_l,reverse=False)
        true_prob_s = None
        true_prob_o = None
        true_prob_r = None
        for cur_t in time_l:
            triples = get_data_with_t(data, cur_t)
            true_s = np.zeros(num_nodes)
            true_o = np.zeros(num_nodes)
            true_r = np.zeros(num_rels)
            s_arr = triples[:,0]
            o_arr = triples[:,2]
            r_arr = triples[:,1]
            for s in s_arr:
                true_s[s] += 1
            for o in o_arr:
                true_o[o] += 1
            for r in r_arr:
                true_r[r] += 1
            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            true_r = true_r / np.sum(true_r)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_nodes)
                true_prob_o = true_o.reshape(1, num_nodes)
                true_prob_r = true_r.reshape(1, num_rels)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_nodes)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_nodes)), axis=0)
                true_prob_r = np.concatenate((true_prob_r, true_r.reshape(1, num_rels)), axis=0)
             
        with open(file_path, 'wb') as fp:
            pickle.dump([true_prob_s,true_prob_r,true_prob_o], fp)
    else:
        print('load true distributions...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [true_prob_s, true_prob_r, true_prob_o] = pickle.load(f)
    true_prob_s = torch.from_numpy(true_prob_s)
    true_prob_r = torch.from_numpy(true_prob_r)
    true_prob_o = torch.from_numpy(true_prob_o)
    return true_prob_s, true_prob_r, true_prob_o 

 

'''
Evaluation metrics
'''
# Label based
 
def print_eval_metrics_Bin(y_true, y_predict, prt=True):
    y_true = [item for sublist in y_true for item in sublist]
    y_predict = [item for sublist in y_predict for item in sublist]
    y_predict = [x>0.5 for x in y_predict]
    
    recall = recall_score(y_true, y_predict, average='binary')
    f1 = f1_score(y_true, y_predict, average='binary')
    beta=2
    f2 = fbeta_score(y_true, y_predict, average='binary', beta=beta)
    hloss = hamming_loss(y_true, y_predict)
    bacc = balanced_accuracy_score(y_true,y_predict)
    acc = accuracy_score(y_true,y_predict)
    if prt:
        print("Rec  weighted: {:.4f}".format(recall))
        print("F1  weighted: {:.4f}".format(f1))
        print("F{}  weighted: {:.4f}".format(beta,f2))
        print("hamming loss: {:.4f}".format(hloss))
        print("bacc: {:.4f}".format(bacc))
        print("acc: {:.4f}".format(acc))
    return hloss, recall, f1, f2, bacc, acc

def print_eval_metrics(true_rank_l, prob_rank_l, prt=True): #?
    m = MultiLabelBinarizer().fit(true_rank_l)
    m_actual = m.transform(true_rank_l)
    m_predicted = m.transform(prob_rank_l)

    p = precision_score(m_actual, m_predicted, average='weighted')
    recall = recall_score(m_actual, m_predicted, average='weighted')
    f1 = f1_score(m_actual, m_predicted, average='weighted')
    beta=2
    f2 = fbeta_score(m_actual, m_predicted, average='weighted', beta=beta)
    hloss = hamming_loss(m_actual, m_predicted)
    #bacc = balanced_accuracy_score(m_actual, m_predicted)
    accuracy_per_l = (m_actual == m_predicted).mean(axis=0)
    acc = accuracy_per_l.mean()
    #acc = accuracy_score(true_rank_l, prob_rank_l)
    if prt:
        print("Pre weighted:{:.4f}".format(p))
        print("Rec  weighted: {:.4f}".format(recall))
        print("F1  weighted: {:.4f}".format(f1))
        print("F{}  weighted: {:.4f}".format(beta,f2))
        print("hamming loss: {:.4f}".format(hloss))
        # print("bacc: {:.4f}".format(bacc))
        print("acc: {:.4f}".format(acc))
    return hloss, p, recall, f1, f2, acc

def print_hit_eval_metrics(total_ranks):
    total_ranks += 1
    mrr = np.mean(1.0 / total_ranks) 
    mr = np.mean(total_ranks)
    hits = []
    for hit in [1, 3, 10]: # , 20, 30
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("Hits @ {}: {:.4f}".format(hit, avg_count))
    # print("MRR: {:.4f} | MR: {:.4f}".format(mrr,mr))
    return hits, mrr, mr

def adaptive_node_drop(g, node_embeds, drop_prob=0.2):
    features = node_embeds[g.ndata['id']]
    node_importance = torch.norm(features, p=2, dim=1)
    threshold = torch.quantile(node_importance, drop_prob)
    
    keep_nodes = [i for i, score in enumerate(node_importance) if score > threshold]
    sg = dgl.node_subgraph(g, keep_nodes)
    return sg

def adaptive_edge_drop(g, drop_prob=0.2):
    num_edges = g.num_edges()
    keep_edges = random.sample(range(num_edges), int(num_edges * (1 - drop_prob)))
    sg = dgl.edge_subgraph(g, keep_edges)
    return sg

def add_noise_to_graphs_gca(g_list,node_embeds):
    noisy_graphs = []
    for g in g_list:
        noisy_g = adaptive_edge_drop(g)
        noisy_g = adaptive_node_drop(noisy_g,node_embeds)
        noisy_graphs.append(noisy_g)
    return noisy_graphs

def add_noise_to_graphs_graphcl(g_list):
    noisy_graphs = []
    for g in g_list:
        noisy_g = augment_graph(g)
        noisy_graphs.append(noisy_g)
    return noisy_graphs

def get_noisy_graph_with_features(g, drop_prob=0.3, add_prob=0.1):
    num_edges = g.num_edges()
    mask = torch.rand(num_edges) > drop_prob
    edge_ids = torch.arange(num_edges)[mask]

    # Clone the original graph
    noisy_g = g.clone()
    noisy_g.remove_edges(torch.arange(num_edges)[~mask])  # Remove edges not in mask
    noisy_g.edata['text_emb'] = g.edata['text_emb'][mask]
    
    return noisy_g

def augment_graph(g, drop_node_prob=0.2):
    num_nodes = g.num_nodes()
    keep_nodes = [i for i in range(num_nodes) if random.random() > drop_node_prob]
    sg = dgl.node_subgraph(g, keep_nodes)
    return sg

def gcl_loss(dp, dg, tau=1.0):
    
    # sim = torch.matmul(dp, dg.transpose(1, 2)) / tau
    # loss = -torch.mean(torch.log(torch.exp(sim) / torch.sum(torch.exp(sim), dim=2, keepdim=True)))

    sim = torch.matmul(dp, dg.transpose(1, 2)) / tau
    log_probs = F.log_softmax(sim, dim=2)
    loss = -torch.mean(log_probs)

    return loss
