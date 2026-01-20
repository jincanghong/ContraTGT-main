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

    # Dataset & Mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use',
                        choices=['socialevolve_1m', 'wiki', 'slashdot', 'bitcoinotc', 'ubuntu'],
                        default='slashdot')

    # Hyper-parameters
    parser.add_argument('--ctx_sample', type=int, default=40, help='spatial sampling')
    parser.add_argument('--tmp_sample', type=int, default=31, help='temporal sampling')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')  # 建议调小 BS 以防显存溢出
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--tolerance', type=float, default=0, help='early stopper tolerance')

    # Computation & Reproducibility
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

    # Model Selection
    parser.add_argument('--model_name', type=str, default='GraphMamba',
                        choices=['ContraTGT', 'CoLA_Former', 'GraphMamba'],
                        help='Choose model architecture')

    parser.add_argument('--suffix', type=str, default='', help='Suffix for saved files')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


class EarlyStopping:
    def __init__(self, save_path, max_round=8, higher_better=True, tolerance=1e-4):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.path = save_path

        dir_name = os.path.dirname(self.path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def __call__(self, curr_val, model):
        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.save_checkpoint(curr_val, model)
            self.last_best = curr_val
            return False

        improvement = (curr_val - self.last_best) / abs(self.last_best)
        if improvement > self.tolerance:
            self.save_checkpoint(curr_val, model)
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            return False

        self.num_round += 1
        if self.num_round >= self.max_round:
            return True
        return False

    def save_checkpoint(self, val_ap, model):
        # print(f'Validation ap improved to {val_ap:.6f}. Saving model...')
        torch.save(model.state_dict(), self.path)
        self.last_best = val_ap


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def Dataset(file='../data/ml_slashdot.csv', starting_line=1):
    if not os.path.exists(file) and os.path.exists(f'../{file}'):
        file = f'../{file}'

    if not os.path.exists(file):
        print(f"Error: Data file not found at {file}")
        sys.exit(1)

    ecols = Namespace({'id': 0, 'FromNodeId': 1, 'ToNodeId': 2, 'TimeStep': 3, 'label': 4, 'idx': 5})

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

    # 简化的 node_time 逻辑，保持原代码兼容性
    node_time = [-1] * (int(nodes_list.max()) + 1)

    idx = edges[:, [ecols.FromNodeId, ecols.ToNodeId, ecols.TimeStep, ecols.idx]]
    labels = edges[:, ecols.label]
    edge = {'idx': idx, 'labels': labels}

    # 使用 70/15/15 划分或原代码的划分
    val_time, test_time = list(np.quantile(edge['idx'][:, 2], [0.70, 0.85]))

    valid_train_flag = (edge['idx'][:, 2] <= val_time)
    valid_val_flag = (edge['idx'][:, 2] > val_time) * (edge['idx'][:, 2] <= test_time)
    valid_test_flag = (edge['idx'][:, 2] > test_time)

    train_data = {'idx': edge['idx'][valid_train_flag], 'labels': edge['labels'][valid_train_flag]}
    val_data = {'idx': edge['idx'][valid_val_flag], 'labels': edge['labels'][valid_val_flag]}
    test_data = {'idx': edge['idx'][valid_test_flag], 'labels': edge['labels'][valid_test_flag]}

    # Inductive Setting
    total_node_set = set(np.array(edge['idx'][:, 0])).union(np.array(edge['idx'][:, 1]))
    train_node_set = set(np.array(train_data['idx'][:, 0])).union(np.array(train_data['idx'][:, 1]))
    new_node_set = total_node_set - train_node_set

    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in
                                 zip(np.array(edge['idx'][:, 0]), np.array(edge['idx'][:, 1]))])

    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    nn_test_data = {'idx': edge['idx'][nn_test_flag], 'labels': edge['labels'][nn_test_flag]}
    nn_val_data = {'idx': edge['idx'][nn_val_flag], 'labels': edge['labels'][nn_val_flag]}

    return edge, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test_data, nn_val_data


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, factor=5, time_encoding='concat'):
        super(TimeEncode, self).__init__()
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float(),
                                             requires_grad=False)
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float(), requires_grad=False)
        self.fc1 = nn.Linear(time_dim * 2, time_dim)
        self.time_encoding = time_encoding
        self.norm = nn.LayerNorm(time_dim, eps=1e-6)

    def forward(self, u, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1) + self.phase.view(1, 1, -1)
        harmonic = self.norm(torch.cos(map_ts))
        if self.time_encoding == "concat":
            x = self.fc1(torch.cat([u, harmonic], dim=-1))
        elif self.time_encoding == "sum":
            x = u * 0.8 + harmonic * 0.2
        return x


def get_edges(s_idx, e_idx, edge):
    batch_edge = edge['idx'][s_idx:e_idx]
    batch_label = edge['labels'][s_idx:e_idx]
    return {'idx': batch_edge, 'labels': batch_label}


# =======================================================
# [核心修改] 严格的确定性种子初始化
# =======================================================
def init_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制使用确定性算法 (可能会稍微降低速度，但保证结果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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