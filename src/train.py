def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import numpy as np
import time
from utils import *
import os
from sklearn.utils import shuffle
from models import *
from data import *
from build_baseline import *
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader 
import json
 
parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="/data3/data/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--model", type=str, default='APEP', help="model name")
parser.add_argument("--n-hidden", type=int, default=32, help="number of hidden units")
parser.add_argument("--ratio", type=float, default=1, help="trainset ratio")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--adversarial_lr", type=float, default=1e-3, help="adversarial learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='EG', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=20, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=7)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--use-gru", type=int, default=1, help='1 use gru 0 rnn')
parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=5, help='number of runs')
parser.add_argument("--n_layers", type=int, default=1, help='number of layers')
parser.add_argument("--text_emd_dim", type=int, default=768, help='text embedding dim')
#baseline
parser = add_baseline_argument(parser)

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()

print("cuda",use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed) 

# eval metrics
recall_list  = []
f1_list  = []
f2_list  = []
hloss_list = []
p_list = []
acc_list = []

iterations = 0 
while iterations < args.runs:  
    iterations += 1
    print('****************** iterations ',iterations,)
    
    if iterations == 1:
        print("loading data...")
        num_nodes, num_rels = utils.get_total_number(
            args.dp + args.dataset, 'stat.txt')
        full_ngh_finder = get_neighbor_finder(f'{args.dp}/{args.dataset}/', 'train.txt', 'valid.txt', 'test.txt', args.uniform, num_nodes)

        train_dataset_loader = DistData( 
            args.dp, args.dataset, args.ratio, num_nodes, num_rels, set_name='train')
        valid_dataset_loader = DistData(
            args.dp, args.dataset, args.ratio, num_nodes, num_rels, set_name='valid')
        test_dataset_loader = DistData(
            args.dp, args.dataset, args.ratio, num_nodes, num_rels, set_name='test')

        #MTG\TGN shuffle=False SeCoGD n_hidden 200 gpu=0 
        
        train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_4)
        valid_loader = DataLoader(valid_dataset_loader, batch_size=1,
                                shuffle=False, collate_fn=collate_4)
        test_loader = DataLoader(test_dataset_loader, batch_size=1,
                                shuffle=False, collate_fn=collate_4)

        #build model
        model = build_model(args, num_nodes, num_rels)
    
        model_name = model.__class__.__name__
        print('Model:', model_name)
        token = '{}_sl{}_max{}_gru{}_attn{}'.format(model_name, args.seq_len, int(args.maxpool), int(args.use_gru),str(args.attn))
        print('Token:', token, args.dataset)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#params:', total_params)

        os.makedirs('models', exist_ok=True)
        os.makedirs('models/' + args.dataset, exist_ok=True)
        model_state_file = 'models/{}/{}.pth'.format(args.dataset, token)
        model_graph_file = 'models/{}/{}_graph.pth'.format(args.dataset, token)
        outf = 'models/{}/{}.result'.format(args.dataset, token)

        if use_cuda:
            model.cuda()

    @torch.no_grad()
    def evaluate(data_loader, dataset_loader, set_name='valid'):
        model.eval()
        true_rank_l = []
        prob_rank_l = []
        total_loss = 0
        data_x = dataset_loader.get_data_x()
        for i, batch in enumerate(tqdm(data_loader)):
            batch_data, true_s, true_r, true_o = batch
            batch_data = torch.stack(batch_data, dim=0)
            if batch_data.item() == 0:
                continue
            true_r = torch.stack(true_r, dim=0)
            true_rank, prob_rank, loss = get_loss(args, batch_data, data_x, true_r, model, set_name)
            #true_rank, prob_rank, loss = model.evaluate(batch_data, true_r)
            true_rank_l.append(true_rank.cpu().tolist())
            prob_rank_l.append(prob_rank.cpu().tolist())
            total_loss += loss.item()
    
        print('{} results'.format(set_name)) 
        hloss, p, recall, f1, f2, acc = utils.print_eval_metrics(true_rank_l,prob_rank_l)
        reduced_loss = total_loss / (dataset_loader.len / 1.0)
        print("{} Loss: {:.6f}".format(set_name, reduced_loss))
        return hloss, p, recall, f1, f2, acc

    def train(data_loader, dataset_loader, set_name='train'):
        train_start_reinitialize(args,model,full_ngh_finder)
        model.train()
        total_loss = 0
        t0 = time.time() 
        data_x = dataset_loader.get_data_x()
        for i, batch in enumerate(tqdm(data_loader)):
            batch_data, true_s, true_r, true_o = batch
            batch_data = torch.stack(batch_data, dim=0)
            if batch_data.item() == 0:
                continue
            true_r = torch.stack(true_r, dim=0)
            loss = get_loss(args, batch_data, data_x, true_r, model, set_name)
            #loss = model(batch_data, true_r)
            with torch.autograd.detect_anomaly():
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            train_end_detach(args,model)
            #clear gpu
            # torch.cuda.empty_cache()
        
        t2 = time.time()
        reduced_loss = total_loss / (dataset_loader.len / args.batch_size)
        print("Epoch {:04d} | Loss {:.6f} | time {:.2f} {}".format(
            epoch, reduced_loss, t2 - t0, time.ctime()))
        return reduced_loss

    bad_counter = 0
    loss_small =  float("inf")
    try:
        print("start training...")
        for epoch in range(1, args.max_epochs+1):
            
            train_loss = train(train_loader, train_dataset_loader, set_name = 'train')
            # evaluate(train_eval_loader, train_dataset_loader, set_name='Train') # eval on train set
            valid_loss, p, recall, f1, f2, acc = evaluate(
                valid_loader, valid_dataset_loader, set_name='valid') # eval on valid set

            if valid_loss < loss_small:
                loss_small = valid_loss
                bad_counter = 0
                print('save better model...')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'global_emb': None}, model_state_file)
                # evaluate(test_loader, test_dataset_loader, set_name='Test')
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        print("training done")

    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early, epoch', epoch)

    # Load the best saved model.
    print("\nstart testing...")
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    hloss, p, recall, f1, f2, acc = evaluate(test_loader, test_dataset_loader, set_name='test')
    print(args)
    recall_list.append(recall)
    f1_list.append(f1)
    f2_list.append(f2)
    hloss_list.append(hloss)
    p_list.append(p)
    acc_list.append(acc)

print('finish training, results ....')
# save average results
recall_list = np.array(recall_list)
f1_list = np.array(f1_list)
f2_list = np.array(f2_list)
hloss_list = np.array(hloss_list)
p_list = np.array(p_list)
acc_list = np.array(acc_list)

recall_avg, recall_std = recall_list.mean(0), recall_list.std(0)
f1_avg, f1_std = f1_list.mean(0), f1_list.std(0)
f2_avg, f2_std = f2_list.mean(0), f2_list.std(0)
hloss_avg, hloss_std = hloss_list.mean(0), hloss_list.std(0)
p_avg, p_std = p_list.mean(0), p_list.std(0)
acc_avg, acc_std = acc_list.mean(0), acc_list.std(0)

print('--------------------')
print("Rec  weighted: {:.4f}".format(recall_avg))
print("F1   weighted: {:.4f}".format(f1_avg))
beta=2
print("F{}  weighted: {:.4f}".format(beta, f2_avg))
print("loss: {:.4f}".format(hloss_avg))
print("pre: {:.4f}".format(p_avg))
print("acc: {:.4f}".format(acc_avg))

# save results
result = 'Model: {}, Dataset: {}, Ratio:{:.4f}, Rec: {:.4f}, Precision: {:.4f}, F1: {:.4f}, F2: {:.4f}, Acc: {:.4f}, Loss: {:.4f}, N_layers: {}, n-hidden: {}, lr: {:.5f}, adversarial_lr: {:.5f}, seq_len: {}, use_gru:{}\n'.format(args.model, args.dataset, args.ratio, recall_avg, p_avg, f1_avg, f2_avg, acc_avg, hloss_avg, args.n_layers, args.n_hidden, args.lr, args.adversarial_lr,args.seq_len,args.use_gru)
with open('/data3/src/results.csv','a') as fd:
    fd.write(result)
