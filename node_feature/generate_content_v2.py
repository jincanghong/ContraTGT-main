import pandas as pd
import numpy as np

def generate_content_v2(csv_path, output_path, dim=127):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    nodes = sorted(list(set(df.u) | set(df.i)))
    num_nodes = len(nodes)
    
    # 使用高斯分布 (mean=0, std=0.01) 生成特征，模拟归一化后的特征分布
    # 这种方式比全零初始化更能体现模型处理连续特征的能力
    np.random.seed(42) # 保证可复现性
    features = np.random.normal(loc=0.0, scale=0.01, size=(num_nodes, dim))
    
    # 构造 DataFrame
    # 按照用户示例，第一列是特征，最后一列可能是 node_id 或者第一列是 node_id
    # 重新观察用户提供的 pasted_content.txt，发现它似乎没有明显的 node_id 列在开头？
    # 或者第一列就是 node_id。通常 .content 第一列是 node_id。
    
    content_data = np.column_stack((nodes, features))
    
    # 保存为 CSV，使用科学计数法格式
    np.savetxt(output_path, content_data, delimiter=',', fmt=['%d'] + ['%.18e'] * dim)
    print(f"Saved {num_nodes} nodes to {output_path} with dim {dim}")

# 根据分析，维度应该是 127
generate_content_v2('/home/ubuntu/upload/ml_slashdot.csv', '/home/ubuntu/slashdot.content', dim=127)
generate_content_v2('/home/ubuntu/ml_ubuntu.csv', '/home/ubuntu/ubuntu.content', dim=127)
