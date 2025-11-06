# plot_informer_prediction.py
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# 配置：修改为你自己的结果目录
# ----------------------------
result_dir = 'results/informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'

# 加载预测和真实值
pred_data = np.load(os.path.join(result_dir, 'pred.npy'))  # shape: [N, pred_len, C]
true_data = np.load(os.path.join(result_dir, 'true.npy'))  # shape: [N, pred_len, C]

print(f"预测形状: {pred_data.shape}")
print(f"真实形状: {true_data.shape}")

# ----------------------------
# 可视化设置
# ----------------------------
num_samples_to_plot = 5  # 绘制前5个样本
pred_len = pred_data.shape[1]  # 预测长度，如24
channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']  # 根据你的数据修改

# 创建时间轴（例如：未来24小时）
time_steps = np.arange(pred_len)

# 绘图
n_vars = pred_data.shape[2]
fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2 * n_vars), sharex=True)
if n_vars == 1:
    axes = [axes]

for var_idx in range(n_vars):
    ax = axes[var_idx]
    # 只绘制第一个样本（你也可以循环多个）
    true_seq = true_data[0, :, var_idx]
    pred_seq = pred_data[0, :, var_idx]

    ax.plot(time_steps, true_seq, label='Ground Truth', color='blue', linewidth=1.5)
    ax.plot(time_steps, pred_seq, label='Prediction', color='red', linestyle='--', linewidth=1.5)
    ax.set_ylabel(channel_names[var_idx] if var_idx < len(channel_names) else f'Var {var_idx}')
    ax.grid(True, linestyle='--', alpha=0.6)
    if var_idx == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Time Steps (Future)')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'plots/prediction_visualization.png'), dpi=300)
plt.show()