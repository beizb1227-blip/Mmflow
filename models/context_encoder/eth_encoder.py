import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from einops import rearrange
import math


class SocialTransformer(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=256, out_dim=128):
        super(SocialTransformer, self).__init__()
        self.encode_past = nn.Linear(in_dim, hidden_dim, bias=False)   # 1. 线性层：将 48 维输入映射到 256 维隐藏层
        # 2. Transformer 编码器层：建模智能体间社交交互（如避让、聚集）
        self.layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.mlp_out = nn.Linear(hidden_dim, out_dim)  # 3. 输出层：将 256 维隐藏特征映射到 128 维，匹配下游 D_MODEL

    def forward(self, past_traj, mask):
        """
        @param past_traj: [B, A, P, D]
        @param mask:      [B, A] or None
        """
        B, A, P, D = past_traj.shape
        # past_traj = rearrange(past_traj, 'b a p d -> (b a) p d')
        # h_feat = self.encode_past(past_traj.reshape(B * A, -1)).unsqueeze(1)  # [B*A, 1, D]

        # 步骤1：将时序轨迹展平为单向量（8帧×6维=48维）→ [B, A, 48]
        past_traj = rearrange(past_traj, 'b a p d -> b a (p d)')
        h_feat = self.encode_past(past_traj)    # [B, A, D]  # 步骤2：线性层映射→[B, A, 256]，提取基础特征

        h_feat_ = self.transformer_encoder(h_feat, mask=mask)   # 步骤3：Transformer 编码→建模智能体间交互（如A1避让A2）→[B, A, 256]

        h_feat = h_feat + h_feat_         # 步骤4：残差连接→缓解梯度消失，增强特征传播
        h_feat = self.mlp_out(h_feat)     # 步骤5：输出层→[B, A, 128]，适配后续 ETHEncoder 的 Transformer 层

        return h_feat
    

class ETHEncoder(nn.Module):
    def __init__(self, config, use_pre_norm):
        super().__init__()
        self.model_cfg = config          # 模型配置（D_MODEL=128、NUM_ATTN_HEAD=8 等）
        dim = self.model_cfg.D_MODEL
        
        ### build social encoder
        self.agent_social_encoder = SocialTransformer(in_dim=48, hidden_dim=256, out_dim=dim)
        
        # Positional encoding    # 2. 位置编码：增强时序/空间位置信息（基于正弦函数，避免位置混淆）
        self.pos_encoding = nn.Sequential(
                SinusoidalPosEmb(dim, theta = 10000),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        # 3. 智能体查询嵌入：为每个智能体分配专属标识（区分不同智能体）
        self.agent_query_embedding = nn.Embedding(self.model_cfg.AGENTS, dim)
        self.mlp_pe = nn.Sequential(         # 4. 位置编码融合 MLP：融合查询嵌入和位置编码→[A, 128]
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # build transformer encoder layers  # 5. Transformer 编码器层：最终特征融合（社交特征+位置特征）
        self.layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                                                nhead=self.model_cfg.NUM_ATTN_HEAD, 
                                                dim_feedforward=dim * 4,   # 前馈网络维度=512，增强特征变换能力
                                                norm_first=use_pre_norm,
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=self.model_cfg.NUM_ATTN_LAYERS)
        self.num_out_channels = dim
        
    ### polyline encoder MLP PointNet [B, A, D]
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder
    

    def forward(self, past_traj):
        """
        Args: [Batch size, Number of agents, Number of time frames, 6]
        """
        # past_traj: [B, A, P, D]  B=批次, A=智能体数量, P=历史时间步, D=轨迹维度(如x,y坐标)
        B, A, P, D = past_traj.shape

        # 1. 社交特征编码：处理每个智能体的历史轨迹，建模智能体间交互
        agent_feature = self.agent_social_encoder(past_traj, mask=None)  # [B, A, D]

        # use positional encoding
        # 2. 位置编码：增强特征的时序/空间位置信息    # 步骤2：生成位置编码→基于智能体数量（A=2）→[A, 128]
        pos_encoding = self.pos_encoding(torch.arange(agent_feature.shape[1]).to(past_traj.device))         # [A, D]

        # enforce positional encoding earlier here
        # 3.智能体查询嵌入：为每个智能体添加专属标识（如编号）
        agent_query = self.agent_query_embedding(torch.arange(self.model_cfg.AGENTS).to(past_traj.device))  # [A, D]

        # 步骤4：融合位置编码和查询嵌入→[A, 128]
        pos_encoding = self.mlp_pe(torch.cat([agent_query, pos_encoding], dim=-1)) # [A, D]
        # 步骤5：特征加位置编码→增强位置信息→[B, A, 128]
        agent_feature += pos_encoding.unsqueeze(0)                              # [B, A, D]
        # 步骤6：最终 Transformer 编码→融合所有特征→[B, A, 128]
        encoder_out = self.transformer_encoder(agent_feature)
        
        return encoder_out 
    