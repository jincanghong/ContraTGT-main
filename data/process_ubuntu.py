import pandas as pd
import gzip

input_file = '/home/ubuntu/sx-askubuntu.txt.gz'
output_file = '/home/ubuntu/ml_ubuntu.csv'

print("Reading and processing data...")
# 原始数据格式: src dst timestamp
# 注意: 原始数据可能包含重复项或未排序，通常时序图数据集需要按时间排序
data = []
with gzip.open(input_file, 'rt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            data.append([int(parts[0]), int(parts[1]), float(parts[2])])

# 转换为 DataFrame
df = pd.DataFrame(data, columns=['u', 'i', 'ts'])

# 按时间戳排序 (时序图学习的重要步骤)
df = df.sort_values(by='ts').reset_index(drop=True)

# 添加 label (默认为 1，表示存在交互)
df['label'] = 1

# 添加 idx (从 1 开始的自增索引)
df['idx'] = range(1, len(df) + 1)

# 调整列顺序以匹配用户要求的格式: ,u,i,ts,label,idx
# 注意: pandas to_csv 默认会带一个未命名的 index 列，正好匹配用户示例开头的逗号
df.to_csv(output_file, index=True)

print(f"Successfully processed {len(df)} interactions.")
print(f"Saved to {output_file}")
print("\nFirst 5 rows of processed data:")
print(df.head())
