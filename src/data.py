import torch
from torch.utils import data
import numpy as np
import utils
import pickle
import collections
from collections import Counter
from event_data_processing import *
# empirical distribution/counts s/r/o in one day, predict r
class DistData(data.Dataset):
      def __init__(self, path, dataset, ratio, num_nodes, num_rels, set_name):
            data, times = utils.load_quadruples(path + dataset, set_name + '.txt')

            if set_name == 'train' and ratio < 1:
                  indices_d = np.random.choice(data.shape[0], size=int(len(data) * ratio), replace=False)
                  data = data[indices_d]
                  indices_t = np.random.choice(times.shape[0], size=int(len(times) * ratio), replace=False)
                  times = times[indices_t]

            data_x = generate_data_x(path + dataset, set_name + '.txt')
            true_prob_s, true_prob_r, true_prob_o = utils.get_true_distributions(path, data, num_nodes, num_rels, dataset, set_name)
            times = torch.from_numpy(times)
            self.data_x = data_x
            self.len = len(times)

            if torch.cuda.is_available():
                  true_prob_s = true_prob_s.cuda()
                  true_prob_r = true_prob_r.cuda()
                  true_prob_o = true_prob_o.cuda()
                  times = times.cuda()

            self.times = times
            self.true_prob_s = true_prob_s
            self.true_prob_r = true_prob_r
            self.true_prob_o = true_prob_o

      def __len__(self):
            return self.len

      def __getitem__(self, index):
            return self.times[index], self.true_prob_s[index], self.true_prob_r[index], self.true_prob_o[index] 
      
      def get_data_x(self): 
            return self.data_x

class DistData_Bin(data.Dataset):
      def __init__(self, path, dataset, target_rels, set_name):
            data, times = utils.load_quadruples(path + dataset, set_name + '.txt')
            y_data = utils.get_y_data(data, target_rels)
            arr = Counter(y_data)
            print(arr)
            
            times = torch.from_numpy(times)
            self.len = len(times)

            y_data = torch.Tensor(y_data)
            if torch.cuda.is_available():
                  y_data = y_data.cuda()
                  times = times.cuda()

            self.times = times
            self.y_data = y_data
            
            
      def __len__(self):
            return self.len

      def __getitem__(self, index):
            return self.times[index], self.y_data[index]


  