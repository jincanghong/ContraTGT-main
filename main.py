from utils import *
import pandas as pd
from sampling import *
import scipy.sparse as sp
import math
import copy
from model import  *
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import random

args, sys_argv = get_args()

GPU = args.gpu
DATA = args.data
LEARNING_RATE = args.lr

device = torch.device('cuda:{}'.format(GPU))

data_name = args.data
edges, num_nodes, nodes_list, node_time,train_data,test_data,val_data,nn_test_data,nn_val_data = Dataset(file = 'data/ml_{}.csv'.format(data_name))
adj_list = get_adj_list(edges)
node_l, ts_l, idx_l, offset_l = init_offset(adj_list)
interaction_list = get_interaction_list(edges)


features = pd.read_csv("node_feature/{}.content".format(data_name),header=None)
features = normalize_features(features)
features = torch.tensor(features)
fea_dim = features.shape[1]
time_encode = TimeEncode(time_dim = fea_dim,time_encoding = 'concat')
time_information = TimeEncode(time_dim = fea_dim,time_encoding = 'sum')

indim = fea_dim
outdim = 128
nheads = 4
dropout = args.drop_out
N = 2
n_epoch = args.n_epoch
BATCH_SIZE = args.bs
num_instance = len(train_data['idx'])
num_batch = math.ceil(num_instance / BATCH_SIZE)
ctx_sample = args.ctx_sample
tmp_sample = args.tmp_sample
pretrain_path = f'pretrain_model/{data_name}.pth'

train_rand_sampler = RandEdgeSampler(train_data['idx'][:,1])
val_rand_sampler = RandEdgeSampler(edges['idx'][:,1])
test_rand_sampler = RandEdgeSampler(edges['idx'][:,1])

st_model = SpatialTemporal(in_dim=indim,out_dim=outdim,n_heads=nheads,dropout=dropout,N=N)
st_model.load_state_dict(torch.load(pretrain_path))
st_model = st_model.to(device)
st_optimizer = optim.Adam(st_model.parameters(),lr=LEARNING_RATE, weight_decay=1e-5)
st_criterion = torch.nn.BCELoss()
st_criterion_eval = torch.nn.BCELoss()
early_stopping = EarlyStopping(dn = data_name, max_round=8)
MODEL_SAVE_PATH = f'saved_models/{data_name}.pth'

def eval_epoch(data, batch_size, model, ctx_sample, tmp_sample, rand_sampler):
    init_seeds(60)
    num_instance = len(data['idx'])
    loss, acc, ap, auc = [], [], [], []
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        model.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            batch_data = get_edges(s_idx, e_idx, data)  
            #             batch_rand_sampler = RandEdgeSampler(batch_data['idx'][:,1])
            node_sum = len(batch_data['idx'])

            # spatial layer
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l,
                                                                                            offset_l,
                                                                                            batch_data['idx'][:, 0],
                                                                                            batch_data['idx'][:, 2],
                                                                                            num_sample=ctx_sample)
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                batch_data['idx'][:, 1],
                                                                                batch_data['idx'][:, 2],
                                                                                num_sample=ctx_sample)

            con_seq_fea = np.empty((node_sum, ctx_sample, indim))  
            con_to_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j]
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j]

            # target node spatial sequence metric
            con_seq_fea = torch.FloatTensor(np.array(con_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts)  
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)

            # dest node spatial sequence metric
            con_to_fea = torch.FloatTensor(np.array(con_to_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts)  
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)

            # ---------------------temporal layer----------------------
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                     interaction_list, flag=True)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]
            # target node temporal sequence metric
            temp_seq_fea = torch.FloatTensor(np.array(temp_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts)  #########
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)

            # dest node temporal sequence metric
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                            interaction_list, flag=False)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts)  
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)

            # fake node spatial and temporal sequence metric
            fake_node = rand_sampler.sample(node_sum)
            fake_con_node, fake_con_ts, fake_con_idx, fake_con_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                        fake_node,
                                                                                        batch_data['idx'][:, 2],
                                                                                        num_sample=ctx_sample)
            fake_con_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_con_node):
                for idj, j in enumerate(i):
                    fake_con_fea[idx, idj, :] = features[j]
            fake_con_fea = torch.FloatTensor(np.array(fake_con_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_con_ts)
            fake_con_fea = time_information(fake_con_fea, ts)
            fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts)).to(device)
            fake_con_mask = torch.LongTensor(np.array(fake_con_mask)).to(device)

            fake_batch_data = copy.deepcopy(batch_data)
            fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
            fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                                interaction_list, flag=False)
            fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(fake_tmp_seq):
                for idj, j in enumerate(i):
                    fake_tmp_fea[idx, idj, :] = features[j]
            fake_tmp_fea = torch.FloatTensor(fake_tmp_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_tmp_ts)
            fake_tmp_fea = time_information(fake_tmp_fea, ts)
            fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts)).to(device)
            fake_temp_mask = torch.LongTensor(np.array(fake_tmp_mask)).to(device)

            pos_label = torch.ones(node_sum, dtype=torch.float, device=device)
            neg_label = torch.zeros(node_sum, dtype=torch.float, device=device)

            pos_prob, neg_prob = st_model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
                                                      batch_node_seq_mask,
                                                      to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                                                      fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)

            st_loss = st_criterion_eval(pos_prob, pos_label)
            st_loss += st_criterion_eval(neg_prob, neg_label)
            loss.append(st_loss.item())

            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(node_sum), np.zeros(node_sum)])
            auc.append(roc_auc_score(true_label, pred_score))
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))

    return np.mean(acc), np.mean(ap), np.average(loss), np.mean(auc)


for m in range(1):
    print('**************************************************')
    print('run {}------ctx:{},tmp:{},dim:{},N:{},dropout:{}'.format(m, ctx_sample, tmp_sample, indim, N, dropout))
    print('**************************************************')

    for epoch in tqdm(range(n_epoch)):
        train_loss, train_acc, train_ap, train_auc = [], [], [], []
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            batch_data = get_edges(s_idx, e_idx, train_data)

            batch_rand_sampler = RandEdgeSampler(batch_data['idx'][:, 1])

            node_sum = len(batch_data['idx'])
            ######################

            # spatial layer
            batch_ngh_node, batch_ngh_ts, batch_ngh_idx, batch_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l,
                                                                                            offset_l,
                                                                                            batch_data['idx'][:, 0],
                                                                                            batch_data['idx'][:, 2],
                                                                                            num_sample=ctx_sample)
            to_ngh_node, to_ngh_ts, to_ngh_idx, to_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                batch_data['idx'][:, 1],
                                                                                batch_data['idx'][:, 2],
                                                                                num_sample=ctx_sample)

            con_seq_fea = np.empty((node_sum, ctx_sample, indim))  
            con_to_fea = np.empty((node_sum, ctx_sample, indim))

            for idx, i in enumerate(batch_ngh_node):
                for idj, j in enumerate(i):
                    con_seq_fea[idx, idj, :] = features[j]
            for idx, i in enumerate(to_ngh_node):
                for idj, j in enumerate(i):
                    con_to_fea[idx, idj, :] = features[j]

            
            con_seq_fea = torch.FloatTensor(np.array(con_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ngh_ts)
            con_seq_fea = time_information(con_seq_fea, ts)  ########
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts)).to(device)
            batch_ngh_mask = torch.LongTensor(np.array(batch_ngh_mask)).to(device)

            
            con_to_fea = torch.FloatTensor(np.array(con_to_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ngh_ts)
            con_to_fea = time_information(con_to_fea, ts)  #########
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts)).to(device)
            to_ngh_mask = torch.LongTensor(np.array(to_ngh_mask)).to(device)

            #################temporal layer###################
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                     interaction_list, flag=True)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]
            
            temp_seq_fea = torch.FloatTensor(np.array(temp_seq_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(batch_ts)
            temp_seq_fea = time_information(temp_seq_fea, ts)  
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts)).to(device)
            batch_node_seq_mask = torch.LongTensor(np.array(batch_node_seq_mask)).to(device)

            
            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                            interaction_list, flag=False)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(to_ts)
            to_seq_fea = time_information(to_seq_fea, ts)
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts)).to(device)
            to_node_seq_mask = torch.LongTensor(np.array(to_node_seq_mask)).to(device)

            ############fake node spatial和temporalseq metric##############
            fake_node = train_rand_sampler.sample(node_sum)
            fake_con_node, fake_con_ts, fake_con_idx, fake_con_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                                        fake_node,
                                                                                        batch_data['idx'][:, 2],
                                                                                        num_sample=ctx_sample)
            fake_con_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_con_node):
                for idj, j in enumerate(i):
                    fake_con_fea[idx, idj, :] = features[j]
            fake_con_fea = torch.FloatTensor(np.array(fake_con_fea))
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_con_ts)
            fake_con_fea = time_information(fake_con_fea, ts)
            fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts)).to(device)
            fake_con_mask = torch.LongTensor(np.array(fake_con_mask)).to(device)

            fake_batch_data = copy.deepcopy(batch_data)
            fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
            fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                                interaction_list, flag=False)
            fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))  
            for idx, i in enumerate(fake_tmp_seq):
                for idj, j in enumerate(i):
                    fake_tmp_fea[idx, idj, :] = features[j]
            fake_tmp_fea = torch.FloatTensor(fake_tmp_fea)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1) - torch.tensor(fake_tmp_ts)
            fake_tmp_fea = time_information(fake_tmp_fea, ts)
            fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts)).to(device)
            fake_temp_mask = torch.LongTensor(np.array(fake_tmp_mask)).to(device)
            ######################train######################
            with torch.no_grad():
                pos_label = torch.ones(node_sum, dtype=torch.float, device=device)
                neg_label = torch.zeros(node_sum, dtype=torch.float, device=device)

            st_optimizer.zero_grad()
            st_model = st_model.train()  ################
            pos_prob, neg_prob = st_model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
                                                      batch_node_seq_mask,
                                                      to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                                                      fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)

            st_loss = st_criterion(pos_prob, pos_label)
            st_loss += st_criterion(neg_prob, neg_label)

            st_loss.backward()
            st_optimizer.step()

            with torch.no_grad():
                st_model = st_model.eval()
                train_loss.append(st_loss.item())
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(node_sum), np.zeros(node_sum)])
                train_acc.append((pred_label == true_label).mean())
                train_ap.append(average_precision_score(true_label, pred_score))
                train_auc.append(roc_auc_score(true_label, pred_score))

        train_loss = np.average(train_loss)
        train_acc = np.mean(train_acc)
        train_ap = np.mean(train_ap)
        train_auc = np.mean(train_auc)

        val_acc, val_ap, val_loss, val_auc = eval_epoch(val_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample,
                                                        val_rand_sampler)
        print('epoch ', epoch, 'train_acc:', train_acc, 'train_ap:', train_ap, 'train_loss:', train_loss, 'train_auc:',
              train_auc)
        print('epoch ', epoch, 'val_acc:', val_acc, 'val_ap:', val_ap, 'val_loss:', val_loss, 'val_auc:', val_auc)

        if early_stopping(val_ap, st_model):
            print("Early stopping")
            best_model_path = f'saved_checkpoints/{data_name}.pth'
            st_model.load_state_dict(torch.load(best_model_path))
            torch.save(st_model.state_dict(), MODEL_SAVE_PATH)
            print("Loaded the best model at epoch {} for inference".format(early_stopping.best_epoch))
            break
    test_acc, test_ap, test_loss, test_auc = eval_epoch(test_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample,
                                                        test_rand_sampler)
    nn_test_acc, nn_test_ap, nn_test_loss, nn_test_auc = eval_epoch(nn_test_data, BATCH_SIZE, st_model, ctx_sample,
                                                                    tmp_sample, test_rand_sampler)

    print('test_auc:', test_auc, 'test_ap:', test_ap, 'test_acc:', test_acc, 'test_loss:', test_loss)
    print('nn_test_auc:', nn_test_auc, 'nn_test_ap:', nn_test_ap, 'nn_test_acc:', nn_test_acc, 'nn_test_loss:',
          nn_test_loss)
