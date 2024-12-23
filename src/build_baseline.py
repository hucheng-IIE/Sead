from models import *
from utils import *
from event_data_processing import *
import torch.nn as nn

def add_baseline_argument(parser):
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to'
                                                                    'backprop')
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
    parser.add_argument('--cache_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of cache updater')
    parser.add_argument('--aggregator', type=str, default="mean", help='Type of message '
                                                                            'aggregator')
    parser.add_argument('--max_pool', type=str, default=True)
    parser.add_argument('--message_dim', type=int, default=32, help='Dimensions of the messages')
    parser.add_argument('--cache_dim', type=int, default=32, help='Dimensions of the cache for '
                                                                    'each user')
    parser.add_argument('--uniform', action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument("--use_gru", type=int, default=1, help='1 use gru 0 rnn')

    parser.add_argument("--k", type=int, default=5, help='number of clusters')
    parser.add_argument("--method", type=str, default='kmeans', help='kmeans,hierarchy,GMM')
    parser.add_argument("--num_s_rels", type=int, default=100, help='number of sample relations')
    parser.add_argument("--disc_func", type=str, default='lin', help='the type of disc_func')
    parser.add_argument("--alpha", type=int, default=0.1, help='loss alpha')
    parser.add_argument("--beta", type=int, default=0.01, help='loss beta')
    parser.add_argument("--agg_mode", type=str, default="GCN", help='agg_mode GCN,SAGEConv,JKNet')
    #secoGD
    parser.add_argument("--encoder", type=str, default="rgcn",help="method of encoder: rgcn/ compgcn")
    parser.add_argument("--decoder", type=str, default="Linear",help="method of decoder")
    # configuration for cross-context hypergraph
    parser.add_argument("--hypergraph_ent", action='store_true', default=False,
                        help="add hypergraph between disentangled nodes")
    parser.add_argument("--hypergraph_rel", action='store_true', default=False,
                        help="add hypergraph between disentangled relations")
    parser.add_argument("--n_layers_hypergraph_ent", type=int, default=1,
                        help="number of propagation rounds on entity hypergraph")
    parser.add_argument("--n_layers_hypergraph_rel", type=int, default=1,
                        help="number of propagation rounds on relation hypergraph")
    parser.add_argument("--score_aggregation", type=str, default='hard',
                        help="score aggregation strategy: hard/ avg")
    parser.add_argument("--k_contexts", type=int, default=5,
                        help="number of contexts to disentangle the sub-embeddings")
    parser.add_argument("--n_hidden", type=int, default=100,
                        help="number of hidden units")
    parser.add_argument("--n_bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--self_loop", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer_norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    return parser

def build_model(args,num_nodes,num_rels):

    with open(args.dp + args.dataset+'/dg_dict.txt', 'rb') as f:
        graph_dict = pickle.load(f)
    print('load dg_dict.txt')

    if args.model == 'Sead':
        model = APEP(h_dim=args.n_hidden, num_ents=num_nodes,
                                num_rels=num_rels, dropout=args.dropout, 
                                seq_len=args.seq_len,
                                maxpool=args.maxpool,
                                use_gru=args.use_gru,
                                attn=args.attn,
                                n_layers=args.n_layers,
                                text_emd_dim=args.text_emd_dim,
                                adversarial_lr=args.adversarial_lr) 
        model.graph_dict = graph_dict

    return model

def train_start_reinitialize(args,model,full_ngh_finder):
    if args.model == 'MTG':
        # Reinitialize cache and memory of the model at the start of each epoch
        model.entity_cache.__init_cache__()
        model.rel_cache.__init_cache__()
        model.memory.__init_memory__()
        # Train using only training graph
        model.set_neighbor_finder(full_ngh_finder)
    elif args.model == 'TGN':
        model.memory.__init_memory__()
def train_end_detach(args,model):
    if args.model == 'MTG':
        model.entity_cache.detach_cache()
        model.rel_cache.detach_cache()
        model.memory.detach_memory()
    elif args.model == 'TGN':
        model.memory.detach_memory()
def get_loss(args, batch_data, all_data_x, y_true, model, mode):
    if mode == 'train': 
        if args.model == 'MTG':
            sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch  = get_batch_data(all_data_x[batch_data])
            y_hat = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, batch_data.item())
            loss = soft_cross_entropy(y_hat, y_true)
        elif args.model == 'glean' or args.model == "CompGCN+RNN" or args.model == "DynamicGCN" or args.model == "TGCN" or args.model == "tRGCN" or args.model == 'SeCoGD' or args.model == 'tGCN':
            y_hat = model.predict(batch_data)
            loss = soft_cross_entropy(y_hat, y_true)
        elif args.model == 'PECF' or args.model == 'CFLP':
            loss, y_hat, embed_F = model.predict(batch_data,y_true)
        elif args.model == 'TGN':
            sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch  = get_batch_data(all_data_x[batch_data])
            NUM_NEIGHBORS = args.n_degree
            y_hat = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
            loss = soft_cross_entropy(y_hat, y_true)
        elif args.model == 'Sead' or args.model == 'GACN' or args.model == 'GraphCL' or args.model == 'GCA':
            loss = model(batch_data,y_true)
        return loss

    elif mode == 'valid':
        if args.model == 'MTG':
            sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch  = get_batch_data(all_data_x[batch_data])
            y_hat_logits = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, batch_data.item())
            true_rank, prob_rank = model.evaluate(y_hat_logits, y_true)
            loss = soft_cross_entropy(y_hat_logits, y_true)
        elif args.model == 'glean' or args.model == "CompGCN+RNN" or args.model == "DynamicGCN" or args.model == "TGCN" or args.model == "tRGCN" or args.model == 'SeCoGD' or args.model == 'tGCN':
            true_rank, prob_rank, loss = model.evaluate(batch_data,y_true)
        elif args.model == 'PECF' or args.model == 'CFLP':
            _, y_hat_logits,_ = model.predict([t_h],y_true) 
        elif args.model == 'TGN':
            NUM_NEIGHBORS = args.n_degree
            sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch  = get_batch_data(all_data_x[batch_data])
            y_hat_logits = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
            true_rank, prob_rank = model.evaluate(y_hat_logits, y_true)
            loss = soft_cross_entropy(y_hat_logits, y_true)
        elif args.model == 'Sead' or args.model == 'GACN' or args.model == 'GraphCL' or args.model == 'GCA':
            true_rank, prob_rank, loss = model.evaluate(batch_data,y_true)
        return true_rank, prob_rank, loss
    
    elif mode == 'test':
        if args.model == 'MTG':
            sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch  = get_batch_data(all_data_x[batch_data])
            y_hat_logits = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, batch_data.item())
            true_rank, prob_rank = model.evaluate(y_hat_logits, y_true)
            loss = soft_cross_entropy(y_hat_logits, y_true)
        elif args.model == 'glean' or args.model == "CompGCN+RNN" or args.model == "DynamicGCN" or args.model == "TGCN" or args.model == "tRGCN" or args.model == 'SeCoGD' or args.model == 'tGCN':
            true_rank, prob_rank, loss = model.evaluate(batch_data,y_true)
        elif args.model == 'PECF' or args.model == 'CFLP':
            _, y_hat_logits,_ = model.predict([t_h],y_true) 
        elif args.model == 'TGN':
            NUM_NEIGHBORS = args.n_degree
            sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch  = get_batch_data(all_data_x[batch_data])
            y_hat_logits = model.predict(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
            true_rank, prob_rank = model.evaluate(y_hat_logits, y_true)
            loss = soft_cross_entropy(y_hat_logits, y_true)
        elif args.model == 'Sead' or args.model == 'GACN' or args.model == 'GraphCL' or args.model == 'GCA':
            true_rank, prob_rank, loss = model.evaluate(batch_data,y_true)
        return true_rank, prob_rank, loss