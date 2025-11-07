import argparse
import torch
import os
import numpy as np

# 假设您将 exp_informer.py 和 data/data_loader.py 放在项目路径中
# 请根据您的实际路径修改导入
from exp.exp_informer import Exp_Informer
from utils.metrics import metric  # 用于在预测后评估性能（可选）


# -----------------------------------------------------------
# 1. 定义与训练时完全一致的参数
# -----------------------------------------------------------
class Args:
    # 核心模型参数 (必须与 Exp1 训练时完全一致)
    model = 'informer'
    data = 'Normal'
    root_path = './data/test/'
    data_path = 'Normal_filtered.csv'
    features = 'MS'
    target = 'Motor Y Voltage'
    freq = 's'
    checkpoints = './checkpoints/'

    # 序列长度参数
    seq_len = 60
    label_len = 30
    pred_len = 20

    # 模型架构参数
    enc_in = 7  # 7 个特征 (8 列 - 1 date = 7)
    dec_in = 7
    c_out = 1  # 预测单变量
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    factor = 5
    attn = 'prob'
    embed = 'timeF'

    # 训练/预测相关参数
    des = 'Exp1'  # <--- 必须是训练时的实验名称
    do_predict = True  # 确保设置为 True
    batch_size = 32
    inverse = False
    mix = True
    distil = True  # 训练时是 True
    output_attention = False

    # 硬件参数 (适应您 Mac 的 MPS 环境)
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'

    # 其他默认参数
    loss = 'mse'
    num_workers = 0
    padding = 0
    activation = 'gelu'
    dropout = 0.05
    s_layers = [3, 2, 1]
    itr = 1
    # 注意: 只需要定义一次，Exp_Informer会自动处理


# -----------------------------------------------------------
# 2. 手动执行预测
# -----------------------------------------------------------
if __name__ == '__main__':
    # 实例化参数和实验类
    args = Args()
    args.detail_freq = args.freq
    Exp = Exp_Informer
    exp = Exp(args)

    setting = 'informer_Normal_ftMS_sl60_ll30_pl20_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_cspFalse_dilateFalse_passthroughFalse_Exp1_0'
    # Informer 内部生成完整 setting 字符串的逻辑
    # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_att{}_fc{}_eb{}_dt{}_{}_{}'.format(
    #     args.model,
    #     args.data,
    #     args.features,
    #     args.seq_len,
    #     args.label_len,
    #     args.pred_len,
    #     args.d_model,
    #     args.n_heads,
    #     args.e_layers,
    #     args.d_layers,
    #     args.d_ff,
    #     args.attn,
    #     args.factor,
    #     args.embed,
    #     args.distil,
    #     args.des,
    #     args.itr)

    print(f"Manual prediction using setting: {setting}")

    # 核心调用: 调用 predict 方法，并设置 load=True
    # 这将手动加载 ./checkpoints/[setting]/checkpoint.pth
    try:
        exp.predict(setting, load=True)
        print("\nPrediction successful. Results saved to ./results/{}/real_prediction.npy".format(setting))

        # 验证测试集性能（可选）
        test_data, test_loader = exp._get_data(flag='test')
        exp.test(setting)
        print("Test complete. Results saved to ./results/{}/pred.npy".format(setting))

    except FileNotFoundError as e:
        print(f"\nERROR: Model checkpoint file not found. Please verify the path:\n{e}")
        print(f"Expected path: {exp.args.checkpoints}/{setting}/checkpoint.pth")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")