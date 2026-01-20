from collections import defaultdict
import numpy as np



#spatial sampling
def get_adj_list(edge):
    adj_list = defaultdict(list)
    for i,edg in enumerate(edge['idx']):
        st = int(edg[0])
        en = int(edg[1])
        ts = int(edg[2])
        idx = int(edg[3])
        adj_list[st].append((en,ts,idx))
        adj_list[en].append((st,ts,idx))
    return adj_list

def init_offset(adj_list):
    node_l = []
    ts_l = []
    idx_l = []
    offset_l = [0]
    for i in range(len(adj_list)):
        curr = adj_list[i]
        curr = sorted(curr, key=lambda x: x[2])
        node_l.extend([x[0] for x in curr])
        ts_l.extend([x[1] for x in curr])
        idx_l.extend([x[2] for x in curr])
        offset_l.append(len(node_l))

    node_l = np.array(node_l)
    ts_l = np.array(ts_l)
    idx_l = np.array(idx_l)
    offset_l = np.array(offset_l)

    return node_l, ts_l, idx_l, offset_l

def find_before(node_l, ts_l, idx_l, offset_l, node, ts):
    ngh_node_l = node_l[offset_l[node]:offset_l[node + 1]]
    ngh_ts_l = ts_l[offset_l[node]:offset_l[node + 1]]
    ngh_idx_l = idx_l[offset_l[node]:offset_l[node + 1]]

    if len(ngh_node_l) == 0 or len(ngh_ts_l) == 0:
        return ngh_node_l, ngh_ts_l, ngh_idx_l

    left = 0
    right = len(ngh_node_l) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        curr_t = ngh_ts_l[mid]
        if curr_t < ts:
            left = mid
        else:
            right = mid

    if ngh_ts_l[right] < ts:
        return ngh_node_l[:right], ngh_idx_l[:right], ngh_ts_l[:right]
    else:
        return ngh_node_l[:left], ngh_idx_l[:left], ngh_ts_l[:left]


def get_neighbor_list(node_l, ts_l, idx_l, offset_l, node, timestamp, num_sample):
    #     print(timestamp)
    ngh_node = np.zeros((len(node), num_sample)).astype(np.int32)
    ngh_ts = np.zeros((len(node), num_sample)).astype(np.int32)
    ngh_idx = np.zeros((len(node), num_sample)).astype(np.int32)
    ngh_mask = np.zeros((len(node), num_sample)).astype(np.int32)
    num_sample = num_sample - 1
    for i, edg in enumerate(zip(node, timestamp)):
        node, idx, ts = find_before(node_l, ts_l, idx_l, offset_l, edg[0], edg[1])
        node = node[::-1]
        idx = idx[::-1]
        ts = ts[::-1]
        #         print(ts)
        #         print('*********************')
        #         print(int(edg[1])-ts)
        if len(node) > 0:
            if len(node) > num_sample:
                node = node[:num_sample]
                idx = idx[:num_sample]
                ts = ts[:num_sample]
            ngh_node[i, 1:len(node) + 1] = node
            ngh_ts[i, 1:len(node) + 1] = int(edg[1]) - ts
            ngh_idx[i, 1:len(node) + 1] = idx
            ngh_mask[i, 1:len(node) + 1] = 1

        ngh_node[i, 0] = edg[0]
        ngh_mask[i, 0] = 1
    return ngh_node, ngh_ts, ngh_idx, ngh_mask

#temporal sampling
def get_interaction_list(edge):
    idx_list = defaultdict(list)
    for i,edg in enumerate(edge['idx']):
        st = int(edg[0])
        en = int(edg[1])
        idx = int(edg[3])
        idx_list[st].append(idx)
        idx_list[en].append(idx)
    return idx_list


def get_unique_node_sequence(b_edge, f_edge, k, idx_list, flag):
    num_nodes = len(b_edge['idx'])
    node_sequence = np.zeros((num_nodes, k)).astype(np.int32)
    node_timestamp = np.zeros((num_nodes, k)).astype(np.int32)
    node_seq_mask = np.zeros((num_nodes, k)).astype(np.int32)
    row = int(k / 2)  # 5
    for i, edg in enumerate(b_edge['idx']):
        ts = int(edg[2])
        idx = int(edg[3])
        # node sequence
        if flag:
            from_node = int(edg[0])
        else:
            from_node = int(edg[1])
        if idx in idx_list[from_node]:
            index = idx_list[from_node].index(idx)
        else:
            left = 0
            right = len(idx_list[from_node]) - 1
            while left + 1 < right:
                mid = (left + right) // 2
                curr_idx = idx_list[from_node][mid]
                if curr_idx < idx:
                    left = mid
                else:
                    right = mid
            index = left
        if index > 0:
            last_event = idx_list[from_node][index - 1]
            if last_event < row:
                node = np.zeros(2 * last_event).astype(np.int32)
                time = np.zeros(2 * last_event).astype(np.int32)
                for j, ed in enumerate(f_edge['idx'][0:last_event]):
                    node[2 * j] = ed[1]
                    node[2 * j + 1] = ed[0]
                    time[2 * j] = ts - int(ed[2])
                    time[2 * j + 1] = ts - int(ed[2])
                node = node[::-1]
                time = time[::-1]
                # print(node)
                node_timestamp[i, 1:2 * last_event + 1] = time
                node_sequence[i, 1:2 * last_event + 1] = node
                node_seq_mask[i, 1:2 * last_event + 1] = 1
            else:
                node = np.zeros(2 * row).astype(np.int32)
                time = np.zeros(2 * row).astype(np.int32)
                for j, ed in enumerate(f_edge['idx'][last_event - row:last_event]):
                    node[2 * j] = ed[1]
                    node[2 * j + 1] = ed[0]
                    time[2 * j] = ts - int(ed[2])
                    time[2 * j + 1] = ts - int(ed[2])
                node = node[::-1]
                time = time[::-1]
                node_timestamp[i, 1:] = time
                node_sequence[i, 1:] = node
                node_seq_mask[i, 1:] = 1
        node_timestamp[i, 0] = 0
        node_sequence[i, 0] = from_node
        node_seq_mask[i, 0] = 1

    return node_sequence, node_seq_mask, node_timestamp