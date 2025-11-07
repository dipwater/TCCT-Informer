import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置参数 ---
DATA_NAME = 'Normal'
FEATURE_NAME = 'S'
# FEATURE_NAME = 'MS'
EXP_NAME = f'informer_{DATA_NAME}_ft{FEATURE_NAME}_sl60_ll30_pl20_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_cspFalse_dilateFalse_passthroughFalse_Exp1_0'  # 您的实验名称
TARGET_NAME = 'Motor Y Voltage'
# 新增：限制绘图的点数
PLOT_LIMIT = 200

# --- 路径设置 ---
RESULTS_DIR = f'./results/{EXP_NAME}/'
PRED_PATH = os.path.join(RESULTS_DIR, 'pred.npy')
TRUE_PATH = os.path.join(RESULTS_DIR, 'true.npy')


# --- 绘图函数 ---
def plot_test_prediction_limited(pred_path, true_path, target_name, limit):
    """加载测试集的预测值和真实值，并限制绘制前 N 个点"""

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"ERROR: One or both files not found in {RESULTS_DIR}")
        print("Expected files: pred.npy and true.npy. Please ensure your training run completed.")
        return

    # 1. 加载数据
    preds_full = np.load(pred_path)
    trues_full = np.load(true_path)

    # 2. 展平数据：形状变为 (Total_Timesteps, Num_Features)
    preds_flat = preds_full.reshape(-1, preds_full.shape[-1])
    trues_flat = trues_full.reshape(-1, trues_full.shape[-1])

    # 由于是 MS 预测，我们只取第一列（Motor Y Voltage）
    preds_data = preds_flat[:, 0]
    trues_data = trues_flat[:, 0]

    # 3. [关键切片操作] 限制到前 N 个点
    if len(preds_data) < limit:
        print(
            f"Warning: Total points available ({len(preds_data)}) is less than the limit ({limit}). Plotting all available points.")
        preds = preds_data
        trues = trues_data
        limit = len(preds_data)
    else:
        preds = preds_data[:limit]
        trues = trues_data[:limit]

    print(f"Total prediction points to plot: {len(preds)} (out of {len(preds_data)} total points in test set)")

    # 4. 创建图表
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(6, 4))
    # plt.figure(figsize=(6, 5))

    # 绘制真实值
    plt.plot(trues, label='true_result', color='#284458', linewidth=2)
    plt.plot(preds, label='pred_result', color='#94ab3a', linewidth=1.5, alpha=0.9)

    # plt.plot(trues, label='true_value', color='blue', alpha=0.7)
    # plt.plot(preds, label='pred_value', color='red', linestyle='-')

    # 添加标签和标题
    plt.ylabel('Motor Y Voltage')
    plt.xlabel('Time(s)')
    # plt.xlabel('Time Step', fontsize=12)
    # plt.ylabel(f'{target_name}', fontsize=12)

    plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper right')
    plt.grid(True, color='#cccccc', linestyle='-', linewidth=0.8)
    # plt.legend()
    # plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # 5. 保存图像
    plot_filename = os.path.join(RESULTS_DIR, f'{DATA_NAME}_{FEATURE_NAME}_true_pred_plot.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    print(f"\nSuccessfully generated plot and saved to: {plot_filename}")
    plt.show()

# --- 运行绘图 ---
plot_test_prediction_limited(PRED_PATH, TRUE_PATH, TARGET_NAME, PLOT_LIMIT)