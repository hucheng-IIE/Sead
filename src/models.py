import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from aggregators import *
from utils import *
from modules_f import *
import time
import math
import random
import itertools
import collections
from Generator import Generator
from Discriminator import Discriminator
import pickle

import logging
#t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

#mtg
from modules.cache import Cache
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.cache_updater import get_cache_updater
from modules.embedding_module import get_embedding_module
#tgn
# from tgn_modules.memory import Memory
# from tgn_modules.message_aggregator import get_message_aggregator
# from tgn_modules.message_function import get_message_function
# from tgn_modules.memory_updater import get_memory_updater
# from tgn_modules.embedding_module import get_embedding_module
 
from seco_modules.decoder import ConvTransE
from seco_modules.aggregator import Aggregator
 
# event forecasting
class APEP(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn='', n_layers=2, text_emd_dim=None, adversarial_lr=0.01):
        super().__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))
 
        #self.word_embeds = None
        self.global_emb = None  
        self.ent_map = None
        self.rel_map = None
        #self.word_graph_dict = None
        self.graph_dict = None
        self.aggregator= aggregator_event_AEPE(h_dim, dropout, num_ents, num_rels, seq_len, maxpool, n_layers,
                                          self.ent_embeds,self.rel_embeds,text_emd_dim)

        #find the most important event
        self.maxpooling = nn.MaxPool2d((num_rels,1))
        self.linear_r = nn.Linear(h_dim, self.num_rels)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = soft_cross_entropy
        self.init_weights()
        
        self.generator = Generator(h_dim, adversarial_lr)
        self.discriminator = Discriminator(h_dim, adversarial_lr)
        #self.gcl_loss = gcl_loss

        self.gcl_mlp = nn.Sequential(
                nn.Linear(h_dim, h_dim),  
                nn.ReLU(),          
                nn.Linear(h_dim, h_dim)   
            )

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def gcl_loss_def(self, z1, z2):                          #(batch_size,num_rels,dim)
        batch_size, num_rels, dim = z1.size()
        g_z1 = self.gcl_mlp(z1)
        g_z2 = self.gcl_mlp(z2)
        #positive sample
        cosine_sim_pos = torch.exp(F.cosine_similarity(g_z1, g_z2, dim=2)) #(batch_size,num_rels)
        #nagtive sample
        cosine_sim_neg = torch.zeros(batch_size,num_rels).cuda()
        
        for i in range(batch_size):
            for j in range(num_rels):
                cosine_sim = torch.exp(F.cosine_similarity(g_z1[i, j].unsqueeze(0), g_z2[i], dim=1))
                cosine_sim_neg[i,j] = cosine_sim.sum()

        sim = torch.log(cosine_sim_pos/cosine_sim_neg)
        loss =  -torch.mean(sim)
        return loss

    def forward(self, t_list, true_prob_r):
        pred_clean, idx_clean, feature_clean = self.__get_pred_embeds(t_list)                            #clean graph
        pred_noise, idx_noise, feature_noise = self.__get_pred_embeds(t_list,add_graph_noise=True)       #adversarial graph
        time = t_list.item()
        # with open('/data3/hucheng/hucheng/IJCAI_2025/src/embedding/embedding_without_gcl'+'_Sead_'+str(time)+'.pkl', 'wb') as f:
        #     pickle.dump(feature_clean, f)
        #adversarial loss
        fake_feature = self.generator(feature_noise)                                                     #adversiarial embedding
        discriminator_loss = self.discriminator.update(feature_clean, fake_feature)
        generator_loss = self.generator.update(self.discriminator, fake_feature)
        gcl_loss = self.gcl_loss_def(feature_clean,fake_feature)
        loss = self.criterion(pred_clean, true_prob_r[idx_clean])

        #sum adversarial loss
        loss = loss + discriminator_loss + generator_loss + gcl_loss

        return loss
 
    def __get_pred_embeds(self, t_list, add_graph_noise=False):
        sorted_t, idx = t_list.sort(0, descending=True)  
        batch_relation_feature = self.aggregator(t_list, self.graph_dict, add_graph_noise)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)

        if torch.cuda.is_available():
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1)).cuda()), dim=0)
        else:
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1))), dim=0)

        #(batch_size,dim)->(batch_size,num_rels)
        pred = self.linear_r(feature_pooling)

        return pred, idx, feature

    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss

        
        #adversarial loss
        fake_feature = self.generator(feature_clean)                                                     #adversiarial embedding
        discriminator_loss = self.discriminator.update(feature_clean, fake_feature)
        generator_loss = self.generator.update(self.discriminator, fake_feature)
        gcl_loss = self.gcl_loss(feature_clean,fake_feature)
        loss = self.criterion(pred_clean, true_prob_r[idx_clean])

        #sum adversarial loss
        loss = loss + discriminator_loss + generator_loss + gcl_loss
        
        return loss
 
    def __get_pred_embeds(self, t_list, add_graph_noise=False):
        sorted_t, idx = t_list.sort(0, descending=True)  
        batch_relation_feature = self.aggregator(t_list, self.graph_dict, add_graph_noise)
        #get relation embeds
        #(batch_size,num_rels,dim)(1,225,32)
        feature = batch_relation_feature.cuda()
        #(batch_size,num_rels,dim)->(batch_size,1,dim)
        feature_pooling = self.maxpooling(feature).squeeze(1)

        if torch.cuda.is_available():
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1)).cuda()), dim=0)
        else:
            feature_pooling = torch.cat((feature_pooling, torch.zeros(len(t_list) - len(feature_pooling), feature_pooling.size(-1))), dim=0)

        #(batch_size,dim)->(batch_size,num_rels)
        pred = self.linear_r(feature_pooling)

        return pred, idx, feature

    def predict(self, t_list, true_prob_r): 
        pred, idx, feature = self.__get_pred_embeds(t_list)
        
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx])
        else:
            loss = None

        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        prob_rel = self.out_func(pred.view(-1))
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss