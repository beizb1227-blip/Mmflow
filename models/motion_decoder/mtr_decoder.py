import copy
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import torch


def modulate(x, shift, scale):
    if len(x.shape) == 3 and len(shift.shape) == 2:
        # [B, K, D] + [B, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 3:
        # [B, K, D] + [B, K, D]
        return x * (1 + scale) + shift
    elif len(x.shape) == 4 and len(shift.shape) == 2:
        # [B, K, A, D] + [B, D]
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 4:
        # [B, K, A, D] + [B, K, A, D]
        return x * (1 + scale) + shift
    else:
        raise ValueError("Invalid shapes to modulate")


class MTRDecoder(nn.Module):
    def __init__(self, config, use_pre_norm, use_adaln=True):
        super().__init__()
        # self.num_blocks = config.get('NUM_DECODER_BLOCKS', 2)
        self.num_blocks = 1
        self.self_attn_K = nn.ModuleList([])         # K-to-K 自注意力（K=预测候选数，配置中=20）
        self.self_attn_A = nn.ModuleList([])         # A-to-A 自注意力（A=智能体数）
        # 模板 Transformer 层：d_model=128，8 头注意力，dropout=0.1
        template_encoder = nn.TransformerEncoderLayer(d_model=config.D_MODEL,  # 128
                                                      dropout=config.get('DROPOUT_OF_ATTN', 0.1),  # 0.1
                                                      nhead=config.NUM_ATTN_HEAD,  # 8
                                                      dim_feedforward=config.D_MODEL * 4,  # 128 * 4
                                                      norm_first=use_pre_norm,  # False
                                                      batch_first=True)
        # import ipdb
        # ipdb.set_trace()
        self.use_adaln = use_adaln    # 时间调制开关（适配 Flow Matching 时间步）

        if use_adaln:
            # 时间调制 MLP：将时间嵌入转换为 shift/scale，用于调制特征
            template_adaln = nn.Sequential(nn.SiLU(),
                                           nn.Linear(config.D_MODEL, 2 * config.D_MODEL, bias=True))

            self.t_adaLN = nn.ModuleList([])

        for _ in range(self.num_blocks):  # 初始化解码块（1 个块包含 K-to-K 和 A-to-A 注意力）
            self.self_attn_K.append(copy.deepcopy(template_encoder))
            self.self_attn_A.append(copy.deepcopy(template_encoder))

            if use_adaln:
                self.t_adaLN.append(copy.deepcopy(template_adaln))

                # zero initialization parameters of adaln
                nn.init.constant_(self.t_adaLN[-1][-1].weight, 0)
                nn.init.constant_(self.t_adaLN[-1][-1].bias, 0)

    def forward(self, query_token, time_emb=None):
        """
        @param query_token: [B, K, A, D] 输入：query_token=[B, K, A, D]（批次×候选数×智能体×128）；time_emb=[B, 128]（时间步嵌入）
                                        输出：[B, K, A, 128] → 解码后的特征（用于生成未来轨迹）
        @param time_emb: [B, D]
        """
        B, K, A = query_token.shape[:3]
        cur_query = query_token

        for i in range(self.num_blocks):
            if self.use_adaln:      # 步骤1：时间调制
                # Flow Matching 中，不同 t 对应 “从纯噪声到真实轨迹” 的不同阶段。shift 和 scale 让模型在每个阶段动态调整特征，更精准地学习 “去噪流”。
                shift, scale = self.t_adaLN[i](time_emb).chunk(2, dim=-1)
                cur_query = modulate(cur_query, shift, scale)  # [B, K, A, D]

            # K-to-K self-attention    # 步骤2：K-to-K 自注意力→建模不同候选轨迹间的依赖（如候选1和候选2的多样性）
            cur_query = rearrange(query_token, 'b k a d -> (b a) k d')
            cur_query = self.self_attn_K[i](cur_query)

            # A-to-A self-attention      # 步骤3：A-to-A 自注意力→建模不同智能体间的依赖（如智能体1和2的协同）
            cur_query = rearrange(cur_query, '(b a) k d -> (b k) a d', b=B)
            cur_query = self.self_attn_A[i](cur_query)

            # reshape
            cur_query = rearrange(cur_query, '(b k) a d -> b k a d', b=B)

        return cur_query     # 输出解码特征，用于后续生成轨迹坐标和置信度

