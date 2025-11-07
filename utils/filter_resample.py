import pandas as pd

# 定义原始文件和输出文件名
input_file = "../data/FLEA/Full.csv"
output_file = input_file.replace('.csv', '_filtered.csv')

# 定义过滤间隔：每隔 5 行取一行
# 注意：我们保留第 1 行，跳过接下来的 4 行，因此使用 step=5
FILTER_STEP = 5

print(f"Loading original data: {input_file}...")

# 1. 读取原始数据，不需要特殊解析，因为我们只做行过滤
# header=0 表示第一行是列名
df = pd.read_csv(input_file, header=0)

# 2. [关键步骤] 使用 iloc 过滤行：从第 0 行开始，到末尾，每隔 FILTER_STEP 取一行
# df.iloc[::5] 会选取索引 0, 5, 10, 15, ... 的行
print(f"Filtering data: taking one row for every {FILTER_STEP} rows...")
df_filtered = df.iloc[::FILTER_STEP].copy()

# 3. 检查结果
print(f"Original shape: {df.shape}")
print(f"Filtered shape: {df_filtered.shape}")

# 4. 保存新的文件
df_filtered.to_csv(output_file, index=False)

print(f"Filtering complete. Saved to {output_file}")
print("Example of first 5 timestamps in filtered data (should be 50ms apart):")
print(df_filtered['date'].head(5))