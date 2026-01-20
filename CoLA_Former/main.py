from utils import *
import pandas as pd
from sampling import *
import scipy.sparse as sp
import math
import copy
from model import *
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import os

# 初始化参数
args, sys_argv = get_args()
GPU = args.gpu
DATA = args.data
LEARNING_RATE = args.lr
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
data_name = args.data

# 路径处理
data_path = f'../data/ml_{data_name}.csv' if not os.path.exists(
    f'data/ml_{data_name}.csv') else f'data/ml_{data_name}.csv'
feat_path = f"../node_feature/{data_name}.content" if not os.path.exists(
    f"node_feature/{data_name}.content") else f"node_feature/{data_name}.content"

# =======================================================
# [核心修改] 生成唯一的文件名，防止多卡运行时覆盖
# =======================================================
if args.suffix:
    run_suffix = args.suffix
else:
    # 默认使用 GPU ID 防止冲突
    run_suffix = f"gpu{args.gpu}"

save_name_base = f"{data_name}_{args.model_name}_{run_suffix}"

# 定义区分后的保存路径
MODEL_SAVE_PATH = f'saved_models/{save_name_base}.pth'
CHECKPOINT_PATH = f'saved_checkpoints/{save_name_base}_ckpt.pth'

print(f"Running Experiment: {args.model_name} on {data_name} (ID: {run_suffix})")
print(f"Final Model will be saved to: {MODEL_SAVE_PATH}")
print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}")
# =======================================================

# 数据加载
edges, num_nodes, nodes_list, node_time, train_data, test_data, val_data, nn_test_data, nn_val_data = Dataset(
    file=data_path)
adj_list = get_adj_list(edges)
node_l, ts_l, idx_l, offset_l = init_offset(adj_list)
interaction_list = get_interaction_list(edges)

features = pd.read_csv(feat_path, header=None)
features = normalize_features(features)
features = torch.tensor(features)
fea_dim = features.shape[1]

# 修复：初始化 TimeEncode 并移动到 device
time_encode = TimeEncode(time_dim=fea_dim, time_encoding='concat').to(device)
time_information = TimeEncode(time_dim=fea_dim, time_encoding='sum').to(device)

# 模型参数
indim = fea_dim
outdim = 256  # Consistent dim
nheads = 4
dropout = args.drop_out
N = 2
n_epoch = args.n_epoch
BATCH_SIZE = args.bs
num_instance = len(train_data['idx'])
num_batch = math.ceil(num_instance / BATCH_SIZE)
ctx_sample = args.ctx_sample
tmp_sample = args.tmp_sample

train_rand_sampler = RandEdgeSampler(train_data['idx'][:, 1])
val_rand_sampler = RandEdgeSampler(edges['idx'][:, 1])
test_rand_sampler = RandEdgeSampler(edges['idx'][:, 1])

# === 模型选择逻辑 ===
if args.model_name == 'CoLA_Former':
    # CoLA-Former 模型
    try:
        from model import CoLA_Former_Model

        st_model = CoLA_Former_Model(in_dim=indim, out_dim=outdim, n_heads=nheads, dropout=dropout, N=N)
    except ImportError:
        # 如果 model.py 还没更新，使用 SpatialTemporal 作为 fallback
        print("Warning: CoLA_Former_Model class not found in model.py, using SpatialTemporal.")
        st_model = SpatialTemporal(in_dim=indim, out_dim=outdim, n_heads=nheads, dropout=dropout, N=N)
else:
    # 原始 ContraTGT 模型
    st_model = SpatialTemporal(in_dim=indim, out_dim=outdim, n_heads=nheads, dropout=dropout, N=N)
    # 尝试加载预训练权重（如果有的话）
    pretrain_path = f'pretrain_model/{data_name}.pth'
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained model from {pretrain_path}")
        st_model.load_state_dict(torch.load(pretrain_path), strict=False)

st_model = st_model.to(device)

# 修复：将 TimeEncode 参数加入优化器 (之前是漏掉的)
st_optimizer = optim.Adam(
    list(st_model.parameters()) + list(time_encode.parameters()) + list(time_information.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

st_criterion = torch.nn.BCELoss()
st_criterion_eval = torch.nn.BCELoss()

# [关键修改] EarlyStopping 传入完整的 CHECKPOINT_PATH
early_stopping = EarlyStopping(save_path=CHECKPOINT_PATH, max_round=8)

# 确保保存目录存在
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_checkpoints', exist_ok=True)


# === 评估函数 ===
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
            node_sum = len(batch_data['idx'])

            # --- Prepare Spatial Data ---
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

            con_seq_fea = torch.FloatTensor(con_seq_fea).to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(batch_ngh_ts).to(device)
            con_seq_fea = time_information(con_seq_fea, ts)
            context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts).to(device))
            batch_ngh_mask = torch.LongTensor(batch_ngh_mask).to(device)

            con_to_fea = torch.FloatTensor(con_to_fea).to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(to_ngh_ts).to(device)
            con_to_fea = time_information(con_to_fea, ts)
            to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts).to(device))
            to_ngh_mask = torch.LongTensor(to_ngh_mask).to(device)

            # --- Prepare Temporal Data ---
            batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                     interaction_list, flag=True)
            temp_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(batch_node_seq):
                for idj, j in enumerate(i):
                    temp_seq_fea[idx, idj, :] = features[j]

            temp_seq_fea = torch.FloatTensor(temp_seq_fea).to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(batch_ts).to(device)
            temp_seq_fea = time_information(temp_seq_fea, ts)
            temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts).to(device))
            batch_node_seq_mask = torch.LongTensor(batch_node_seq_mask).to(device)

            to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                            interaction_list, flag=False)
            to_seq_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(to_node_seq):
                for idj, j in enumerate(i):
                    to_seq_fea[idx, idj, :] = features[j]
            to_seq_fea = torch.FloatTensor(to_seq_fea).to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(to_ts).to(device)
            to_seq_fea = time_information(to_seq_fea, ts)
            to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts).to(device))
            to_node_seq_mask = torch.LongTensor(to_node_seq_mask).to(device)

            # --- Prepare Fake Data ---
            fake_node = rand_sampler.sample(node_sum)
            fake_con_node, fake_con_ts, _, fake_con_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                             fake_node,
                                                                             batch_data['idx'][:, 2],
                                                                             num_sample=ctx_sample)
            fake_con_fea = np.empty((node_sum, ctx_sample, indim))
            for idx, i in enumerate(fake_con_node):
                for idj, j in enumerate(i):
                    fake_con_fea[idx, idj, :] = features[j]
            fake_con_fea = torch.FloatTensor(fake_con_fea).to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(fake_con_ts).to(device)
            fake_con_fea = time_information(fake_con_fea, ts)
            fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts).to(device))
            fake_con_mask = torch.LongTensor(fake_con_mask).to(device)

            fake_batch_data = copy.deepcopy(batch_data)
            fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
            fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                                interaction_list, flag=False)
            fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))
            for idx, i in enumerate(fake_tmp_seq):
                for idj, j in enumerate(i):
                    fake_tmp_fea[idx, idj, :] = features[j]
            fake_tmp_fea = torch.FloatTensor(fake_tmp_fea).to(device)
            ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(fake_tmp_ts).to(device)
            fake_tmp_fea = time_information(fake_tmp_fea, ts)
            fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts).to(device))
            fake_temp_mask = torch.LongTensor(fake_tmp_mask).to(device)

            # --- Prediction ---
            pos_label = torch.ones(node_sum, dtype=torch.float, device=device)
            neg_label = torch.zeros(node_sum, dtype=torch.float, device=device)

            pos_prob, neg_prob = model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
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


# === 训练循环 ===
print('**************************************************')
print(f'Training {args.model_name} on {data_name}...')
print('**************************************************')

for epoch in tqdm(range(n_epoch)):
    train_loss, train_acc, train_ap, train_auc = [], [], [], []
    for k in range(num_batch):
        # ... (数据准备部分，为了节省篇幅，这里复用上面的逻辑，实际运行请确保包含完整的数据准备代码) ...
        # 简单起见，这里假设你已经把 Eval 中的数据准备代码复制过来了
        # ...

        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        batch_data = get_edges(s_idx, e_idx, train_data)
        node_sum = len(batch_data['idx'])

        # --- Spatial Data ---
        batch_ngh_node, batch_ngh_ts, _, batch_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                            batch_data['idx'][:, 0],
                                                                            batch_data['idx'][:, 2],
                                                                            num_sample=ctx_sample)
        to_ngh_node, to_ngh_ts, _, to_ngh_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
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

        con_seq_fea = torch.FloatTensor(con_seq_fea).to(device)
        ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(batch_ngh_ts).to(device)
        con_seq_fea = time_information(con_seq_fea, ts)
        context_feature = time_encode(con_seq_fea, torch.tensor(batch_ngh_ts).to(device))
        batch_ngh_mask = torch.LongTensor(batch_ngh_mask).to(device)

        con_to_fea = torch.FloatTensor(con_to_fea).to(device)
        ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(to_ngh_ts).to(device)
        con_to_fea = time_information(con_to_fea, ts)
        to_con_feature = time_encode(con_to_fea, torch.tensor(to_ngh_ts).to(device))
        to_ngh_mask = torch.LongTensor(to_ngh_mask).to(device)

        # --- Temporal Data ---
        batch_node_seq, batch_node_seq_mask, batch_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                                 interaction_list, flag=True)
        temp_seq_fea = np.empty((node_sum, tmp_sample, indim))
        for idx, i in enumerate(batch_node_seq):
            for idj, j in enumerate(i):
                temp_seq_fea[idx, idj, :] = features[j]

        temp_seq_fea = torch.FloatTensor(temp_seq_fea).to(device)
        ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(batch_ts).to(device)
        temp_seq_fea = time_information(temp_seq_fea, ts)
        temporal_feature = time_encode(temp_seq_fea, torch.tensor(batch_ts).to(device))
        batch_node_seq_mask = torch.LongTensor(batch_node_seq_mask).to(device)

        to_node_seq, to_node_seq_mask, to_ts = get_unique_node_sequence(batch_data, edges, tmp_sample,
                                                                        interaction_list, flag=False)
        to_seq_fea = np.empty((node_sum, tmp_sample, indim))
        for idx, i in enumerate(to_node_seq):
            for idj, j in enumerate(i):
                to_seq_fea[idx, idj, :] = features[j]
        to_seq_fea = torch.FloatTensor(to_seq_fea).to(device)
        ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(to_ts).to(device)
        to_seq_fea = time_information(to_seq_fea, ts)
        to_seq_feature = time_encode(to_seq_fea, torch.tensor(to_ts).to(device))
        to_node_seq_mask = torch.LongTensor(to_node_seq_mask).to(device)

        # --- Fake Data ---
        fake_node = train_rand_sampler.sample(node_sum)
        fake_con_node, fake_con_ts, _, fake_con_mask = get_neighbor_list(node_l, ts_l, idx_l, offset_l,
                                                                         fake_node,
                                                                         batch_data['idx'][:, 2],
                                                                         num_sample=ctx_sample)
        fake_con_fea = np.empty((node_sum, ctx_sample, indim))
        for idx, i in enumerate(fake_con_node):
            for idj, j in enumerate(i):
                fake_con_fea[idx, idj, :] = features[j]
        fake_con_fea = torch.FloatTensor(fake_con_fea).to(device)
        ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(fake_con_ts).to(device)
        fake_con_fea = time_information(fake_con_fea, ts)
        fake_con_fea = time_encode(fake_con_fea, torch.tensor(fake_con_ts).to(device))
        fake_con_mask = torch.LongTensor(fake_con_mask).to(device)

        fake_batch_data = copy.deepcopy(batch_data)
        fake_batch_data['idx'][:, 1] = torch.tensor(fake_node)
        fake_tmp_seq, fake_tmp_mask, fake_tmp_ts = get_unique_node_sequence(fake_batch_data, edges, tmp_sample,
                                                                            interaction_list, flag=False)
        fake_tmp_fea = np.empty((node_sum, tmp_sample, indim))
        for idx, i in enumerate(fake_tmp_seq):
            for idj, j in enumerate(i):
                fake_tmp_fea[idx, idj, :] = features[j]
        fake_tmp_fea = torch.FloatTensor(fake_tmp_fea).to(device)
        ts = batch_data['idx'][:, 2].unsqueeze(dim=-1).to(device) - torch.tensor(fake_tmp_ts).to(device)
        fake_tmp_fea = time_information(fake_tmp_fea, ts)
        fake_temp_fea = time_encode(fake_tmp_fea, torch.tensor(fake_tmp_ts).to(device))
        fake_temp_mask = torch.LongTensor(fake_tmp_mask).to(device)

        # Forward
        st_optimizer.zero_grad()
        st_model = st_model.train()

        pos_label = torch.ones(node_sum, dtype=torch.float, device=device)
        neg_label = torch.zeros(node_sum, dtype=torch.float, device=device)

        pos_prob, neg_prob = st_model.linkPredict(context_feature, batch_ngh_mask, temporal_feature,
                                                  batch_node_seq_mask,
                                                  to_con_feature, to_ngh_mask, to_seq_feature, to_node_seq_mask,
                                                  fake_con_fea, fake_con_mask, fake_temp_fea, fake_temp_mask)

        st_loss = st_criterion(pos_prob, pos_label) + st_criterion(neg_prob, neg_label)
        st_loss.backward()
        st_optimizer.step()

        # Metrics
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

    print(
        f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

    if early_stopping(val_ap, st_model):
        print("Early stopping")
        # [Fix] 加载时也使用 Checkpoint Path
        st_model.load_state_dict(torch.load(early_stopping.path))
        torch.save(st_model.state_dict(), MODEL_SAVE_PATH)
        break

# Final Testing
test_acc, test_ap, test_loss, test_auc = eval_epoch(test_data, BATCH_SIZE, st_model, ctx_sample, tmp_sample,
                                                    test_rand_sampler)
nn_test_acc, nn_test_ap, nn_test_loss, nn_test_auc = eval_epoch(nn_test_data, BATCH_SIZE, st_model, ctx_sample,
                                                                tmp_sample, test_rand_sampler)

print('--------------------------------------------------')
print(f'Test Results ({args.model_name}):')
print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Test ACC: {test_acc:.4f}')
print(f'New Node Test AUC: {nn_test_auc:.4f}')
print('--------------------------------------------------')