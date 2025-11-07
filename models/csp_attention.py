import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CSPAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_half = d_model // 2
        self.q_linear = nn.Linear(d_half, d_half)
        self.k_linear = nn.Linear(d_half, d_half)
        self.v_linear = nn.Linear(d_half, d_half)
        self.out_linear = nn.Linear(d_half, d_half)
        self.conv = nn.Linear(d_half, d_half)  # 1x1 conv
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        part1, part2 = torch.split(x, self.d_model // 2, dim=2)
        part1 = self.conv(part1)

        q = self.q_linear(part2)
        k = self.k_linear(part2)
        v = self.v_linear(part2)

        q = q.view(q.size(0), q.size(1), self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), self.n_heads, -1).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.size(0), out.size(1), -1)
        out = self.out_linear(out)

        result = torch.cat((part1, out), dim=2)
        return result


class DilatedCausalConv(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        padding = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        x = F.pad(x, (padding, 0))  # 因果填充
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class TCCTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList([CSPAttention(d_model, n_heads, dropout) for _ in range(num_layers)])
        self.distill_layers = nn.ModuleList(
            [DilatedCausalConv(d_model, dilation=2 ** i) for i in range(num_layers - 1)])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.transition = nn.Linear(d_model * (2 ** num_layers - 1), d_model)

    def forward(self, x, attn_mask=None):
        outputs = []
        for i in range(self.num_layers):
            x = self.attn_layers[i](x, attn_mask)
            outputs.append(x)
            if i < self.num_layers - 1:
                x = self.distill_layers[i](x)
                x = self.maxpool(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 直通机制
        n = self.num_layers
        min_len = outputs[-1].size(1)
        fused = []
        for k in range(n):
            out = outputs[k]
            split_num = 2 ** (n - k - 1)
            sub_maps = torch.chunk(out, split_num, dim=1) if split_num > 1 else [out]
            fused.extend([sm[:, -min_len:, :] for sm in sub_maps])  # 调整到最小长度
        fused = torch.cat(fused, dim=2)
        fused = self.transition(fused)
        return fused


class SimpleDecoder(nn.Module):
    def __init__(self, d_model, output_size):
        super().__init__()
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):
        return self.linear(x)


class TCCTModel(nn.Module):
    def __init__(self, d_model=512, n_heads=8, num_layers=3, dropout=0.1, output_size=1):
        super().__init__()
        self.encoder = TCCTEncoder(d_model, n_heads, num_layers, dropout)
        self.decoder = SimpleDecoder(d_model, output_size)

    def forward(self, x, attn_mask=None):
        enc_out = self.encoder(x, attn_mask)
        out = self.decoder(enc_out)
        return out

# 示例使用
# model = TCCTModel()
# input = torch.rand(32, 96, 512)  # batch, seq, d_model
# output = model(input)
# print(output.shape)  # (32, seq/4 or similar, 1)