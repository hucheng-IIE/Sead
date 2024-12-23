import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, n_rels, memory_dimension, input_dimension,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.n_rels = n_rels
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    # self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.W_mem_node = torch.nn.Linear(input_dimension, memory_dimension)
    self.W_mem_rel = torch.nn.Linear(input_dimension, memory_dimension)

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.time = 0
    self.entity_memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.rel_memory = nn.Parameter(torch.zeros((self.n_rels, self.memory_dimension)).to(self.device),
                               requires_grad=False)

  def detach_memory(self):
    self.entity_memory.detach_()
    self.rel_memory.detach_()
    # Detach all stored messages

  def update_memory(self, nodes_embeddings, rels_embeddings, nodes_ids , rels_ids):
      if self.time > 1:
          self.entity_memory[:]  = torch.div(self.entity_memory, (self.time+1)/self.time)
          self.rel_memory[:]  = torch.div(self.rel_memory ,(self.time+1)/self.time)
      nodes_new_memory = self.W_mem_node(nodes_embeddings)
      # print (self.entity_memory[nodes_ids].shape)
      # print (torch.div(nodes_new_memory,(self.time+1)).shape)
      self.entity_memory[nodes_ids] += torch.div(nodes_new_memory,(self.time+1))
      rels_new_memory = self.W_mem_rel(rels_embeddings)
      self.rel_memory[rels_ids] += torch.div(rels_new_memory,(self.time+1))
      self.time += 1

  def get_nodes_memory(self, nodes_ids):
      return self.entity_memory[nodes_ids]


  def get_rels_memory(self, rels_ids):
      return self.rel_memory[rels_ids]
