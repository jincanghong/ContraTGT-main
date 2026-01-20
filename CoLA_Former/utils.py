import numpy as np
import torch
import torch.nn as nn
import os
import random
import argparse
import sys
import scipy.sparse as sp


def get_args():
    parser = argparse.ArgumentParser(
        'Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        choices=['socialevolve_1m', 'wiki', 'slashdot', 'bitcoinotc', 'ubuntu'],
                        default='slashdot')

    # general training hyper-parameters
    parser.add_argument('--ctx_sample', type=int, default=40, help='spatial sampling')
    parser.add_argument('--tmp_sample', type=int, default=31, help='temporal sampling')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=800, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0,
                        help='tolerated marginal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=60, help='random seed for all randomized algorithms')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--aug_len', type=float, default=1.5, help='augmentation seq lenqth')

    # === 新增参数 ===
    parser.add_argument('--model_name', type=str, default='CoLA_Former',
                        choices=['ContraTGT', 'CoLA_Former'],
                        help='Choose the model architecture for experiment')

    # [关键修改] 添加 suffix 参数，用于区分不同实验跑路 (Run ID)
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for saved files (e.g. run1, gpu0). If empty, uses gpu id automatically.')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


class EarlyStopping:
    # [关键修改] __init__ 接收 save_path 而不是只接收 dn
    def __init__(self, save_path, max_round=8, higher_better=True, tolerance=1e-4):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.path = save_path  # 直接使用传入的完整路径
        self.best_loss_history = []

        # 自动确保目录存在
        dir_name = os.path.dirname(self.path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def __call__(self, curr_val, model):
        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.save_checkpoint(curr_val, model)
            self.last_best = curr_val
            self.best_loss_history.append(curr_val)
            return False

        improvement = (curr_val - self.last_best) / abs(self.last_best)
        if improvement > self.tolerance:
            self.save_checkpoint(curr_val, model)
            self.last_best = curr_val
            self.best_loss_history.append(curr_val)
            self.num_round = 0
            self.best_epoch = self.epoch_count
            if len(self.best_loss_history) > 10:
                self.best_loss_history.pop(0)
            return False

        self.num_round += 1
        if self.num_round >= self.max_round:
            print(f"Early stopping triggered at epoch {self.epoch_count}")
            return True
        else:
            return False

    def save_checkpoint(self, val_ap, model):
        # [Fix] 修复 None 打印报错，并打印保存路径
        if self.last_best is None:
            print(f'Validation ap improved (First Record: {val_ap:.6f}). Saving model to {self.path} ...')
        else:
            print(
                f'Validation ap improved (from {self.last_best:.6f} to {val_ap:.6f}). Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.last_best = val_ap


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def Dataset(file='../data/ml_slashdot.csv', starting_line=1):
    # 路径兼容处理
    if not os.path.exists(file) and os.path.exists(f'../{file}'):
        file = f'../{file}'

    ecols = Namespace({'id': 0,
                       'FromNodeId': 1,
                       'ToNodeId': 2,
                       'TimeStep': 3,
                       'label': 4,
                       'idx': 5
                       })

    with open(file) as f:
        lines = f.read().splitlines()
    edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]
    edges = torch.tensor(edges, dtype=torch.long)

    new_edges = edges[:, [ecols.FromNodeId, ecols.ToNodeId]]
    _, new_edges = new_edges.unique(return_inverse=True)
    edges[:, [ecols.FromNodeId, ecols.ToNodeId]] = new_edges
    timestamp = edges[:, [ecols.TimeStep]]

    num_nodes = edges[:, [ecols.FromNodeId, ecols.ToNodeId]].unique().size(0)
    nodes_list = edges[:, [ecols.FromNodeId, ecols.ToNodeId]].unique()

    node_time = [-1] * num_nodes
    for i, edg in enumerate(edges):
        st = int(edg[ecols.FromNodeId])
        en = int(edg[ecols.ToNodeId])
        if node_time[st] == -1:
            node_time[st] = int(timestamp[i])
        if node_time[en] == -1:
            node_time[en] = int(timestamp[i])

    idx = edges[:, [ecols.FromNodeId,
                    ecols.ToNodeId,
                    ecols.TimeStep,
                    ecols.idx]]
    labels = edges[:, ecols.label]

    edge = {'idx': idx, 'labels': labels}

    # val_time, test_time = list(np.quantile(edge['idx'][:, 2], [0.10, 0.20]))
    val_time, test_time = list(np.quantile(edge['idx'][:, 2], [0.70, 0.85]))
    valid_train_flag = (edge['idx'][:, 2] <= val_time)
    valid_val_flag = (edge['idx'][:, 2] > val_time) * (edge['idx'][:, 2] <= test_time)
    valid_test_flag = (edge['idx'][:, 2] > test_time)

    train_edge = edge['idx'][valid_train_flag]
    train_label = edge['labels'][valid_train_flag]
    train_data = {'idx': train_edge, 'labels': train_label}
    test_edge = edge['idx'][valid_test_flag]
    test_label = edge['labels'][valid_test_flag]
    test_data = {'idx': test_edge, 'labels': test_label}
    val_edge = edge['idx'][valid_val_flag]
    val_label = edge['labels'][valid_val_flag]
    val_data = {'idx': val_edge, 'labels': val_label}

    total_node_set = set(np.array(edge['idx'][:, 0])).union(np.array(edge['idx'][:, 1]))
    train_node_set = set(np.array(train_data['idx'][:, 0])).union(np.array(train_data['idx'][:, 1]))
    new_node_set = total_node_set - train_node_set

    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in
                                 zip(np.array(edge['idx'][:, 0]), np.array(edge['idx'][:, 1]))])

    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    nn_test_edge = edge['idx'][nn_test_flag]
    nn_test_label = edge['labels'][nn_test_flag]
    nn_test_data = {'idx': nn_test_edge, 'labels': nn_test_label}
    nn_val_edge = edge['idx'][nn_val_flag]
    nn_val_label = edge['labels'][nn_val_flag]
    nn_val_data = {'idx': nn_val_edge, 'labels': nn_val_label}

    return edge, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test_data, nn_val_data


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, factor=5, time_encoding='concat'):
        super(TimeEncode, self).__init__()
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float(),
                                             requires_grad=False)
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float(), requires_grad=False)
        self.fc1 = nn.Linear(time_dim * 2, time_dim)
        self.time_encoding = time_encoding
        self.act = nn.LeakyReLU()
        self.norm = nn.LayerNorm(time_dim, eps=1e-6)

    def forward(self, u, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = self.norm(torch.cos(map_ts))
        if self.time_encoding == "concat":
            x = self.fc1(torch.cat([u, harmonic], dim=-1))
        elif self.time_encoding == "sum":
            x = u * 0.8 + harmonic * 0.2
        return x


def get_edges(s_idx, e_idx, edge):
    batch_edge = edge['idx'][s_idx:e_idx]
    batch_label = edge['labels'][s_idx:e_idx]
    batch_data = {'idx': batch_edge, 'labels': batch_label}
    return batch_data


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class RandEdgeSampler(object):
    def __init__(self, dst_list):
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.dst_list[dst_index]


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
