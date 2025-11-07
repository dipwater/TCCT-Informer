import pandas as pd

input_file = "../data/FLEA/Normal.csv"
print("Loading data...")
# 1. 读取数据，并将 'date' 列解析为时间戳
df = pd.read_csv(input_file, parse_dates=['date'])

# 2. 将 'date' 列设为索引，这是重采样的前提
df.set_index('date', inplace=True)

# sample_name = '1s'
resample_name = '50ms'
# 3. [关键] 重采样到 1 秒 (1S) 频率
#    我们将 100 个 10ms 的点聚合成 1 个点
#    使用 .mean() (平均值) 是最常见的做法
print(f"Resampling data to {resample_name} frequency...")
df_resampled = df.resample(resample_name).mean()

# 4. 去除因重采样可能产生的空值 (NaN)
df_resampled.dropna(inplace=True)

# 5. [重要] Informer 需要一个 'date' 列，而不是索引
df_resampled.reset_index(inplace=True)

# 6. 保存新的、可用于 Informer 的文件
new_filename = input_file.replace('.csv', f'_{resample_name}_resampled.csv')
df_resampled.to_csv(new_filename, index=False)

print(f"Resampling complete. Saved to {new_filename}")
print("Original shape:", df.shape)
print("New shape:", df_resampled.shape)