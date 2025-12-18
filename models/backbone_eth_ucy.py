import numpy as np

import torch
import torch.nn as nn
from .context_encoder import build_context_encoder
from .motion_decoder import build_decoder
from .motion_decoder.mtr_decoder import modulate
from .utils.common_layers import build_mlps
from einops import repeat, rearrange
from models.context_encoder.mtr_encoder import SinusoidalPosEmb


class ETHMotionTransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.config = config

        use_pre_norm = self.model_cfg.get('USE_PRE_NORM', False)

        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm)

        ### serves the purpose of positional encoding
        self.motion_query_embedding = nn.Embedding(self.model_cfg.NUM_PROPOSED_QUERY, self.dim)
        self.agent_order_embedding = nn.Embedding(self.model_cfg.CONTEXT_ENCODER.AGENTS, self.dim)
        self.post_pe_cat_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        time_dim = self.dim * 1
        sinu_pos_emb = SinusoidalPosEmb(self.dim, theta = 10000)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim // 2),
        )
        self.r_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim // 2),
        )

        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(self.model_cfg.MODEL_OUT_DIM, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        dropout_ = self.model_cfg.MOTION_DECODER.DROPOUT_OF_ATTN
        self.noisy_y_attn_k = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True)
        self.noisy_y_attn_a = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True)

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(self.dim + time_dim + self.dim, self.dim),
            #nn.Linear(time_dim + self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(dim_decoder, dim_decoder),
            nn.ReLU(),
            nn.Linear(dim_decoder, self.model_cfg.MODEL_OUT_DIM),
        )

        self.motion_decoder = build_decoder(self.model_cfg.MOTION_DECODER, use_pre_norm)

        self.reg_head = build_mlps(c_in=self.dim, mlp_channels=self.model_cfg.REGRESSION_MLPS, ret_before_act=True, without_norm=True)
        self.cls_head = build_mlps(c_in=dim_decoder, mlp_channels=self.model_cfg.CLASSIFICATION_MLPS, ret_before_act=True, without_norm=True)

        # print out the number of parameters
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_decoder = sum(p.numel() for p in self.motion_decoder.parameters())
        params_total = sum(p.numel() for p in self.parameters())
        params_other = params_total - params_encoder - params_decoder
        logger.info("Total parameters: {:,}, Encoder: {:,}, Decoder: {:,}, Other: {:,}".format(params_total, params_encoder, params_decoder, params_other))

    def forward(self, y, time, r, x_data):
        """
        y: noisy vector
        time: denoising time step
        x_data: data dict 
        """

        ### ETH_UCY assertions
        assert y.shape[-1] == 24, 'y shape is not correct'
        device = y.device
        B, K, A, _ = y.shape

        ### context encoder 上下文
        encoder_out = self.context_encoder(x_data['past_traj_original_scale'])  # [B, A, D]
        encoder_out_batch = repeat(encoder_out, 'b a d -> b k a d', k=K, a=A) 	# [B, K, A, D]

        ### init embeddings

        y_emb = self.noisy_y_mlp(y)  	# [B, K, A, D]

        time_ = time
        r_ = r
        if self.config.denoising_method == 'fm':
            time = time * 1000.0  # flow matching time upscaling
            r = r * 1000.0
        # 时间编码
        t_emb = self.time_mlp(time) 	# [B, D] # (B,1)->(B,64)

        t_emb_batch = repeat(t_emb, 'b d -> b k a d', b=B, k=K, a=A) # [B, K, A, D] batch size , K = 20 , agent = 1 /2 /3 ,dim
        r_emb = self.r_mlp(r) 	# [B, D]
        r_emb_batch = repeat(r_emb, 'b d -> b k a d', b=B, k=K, a=A) # [B, K, A, D]
        t_emb = torch.cat((t_emb, r_emb), dim=-1)
        t_emb_batch = torch.cat((t_emb_batch, r_emb_batch), dim=-1)

        # 1. 预测候选（K）的位置嵌入(区分不同候选轨迹)：区分不同的预测候选（如“第1个候选”和“第2个候选”）
        k_pe = self.motion_query_embedding(torch.arange(K, device=device))	# [K, D]
        k_pe_batch = repeat(k_pe, 'k d -> b k a d', b=B, a=A)	# [B, K, A, D]
        # 2. 智能体（A）的位置嵌入：区分不同的智能体（如“行人1”和“行人2”）
        a_pe = self.agent_order_embedding(torch.arange(A, device=device))  # [A, D]
        a_pe_batch = repeat(a_pe, 'a d -> b k a d', b=B, k=K)	# [B, K, A, D]

        # 将位置嵌入添加到噪声特征中
        y_emb = y_emb + k_pe_batch + a_pe_batch
        y_emb_k = rearrange(y_emb, 'b k a d -> (b a) k d')        #[2560,20,1,28]
        y_emb_k = self.noisy_y_attn_k(y_emb_k)
        y_emb = rearrange(y_emb_k, '(b a) k d -> b k a d', b=B, a=A)

        y_emb_a = rearrange(y_emb, 'b k a d -> (b k) a d')         #[2560,20,1,28]
        y_emb_a = self.noisy_y_attn_a(y_emb_a)
        y_emb = rearrange(y_emb_a, '(b k) a d -> b k a d', b=B, k=K)

        #动态丢弃特征模拟噪声，迫使模型学习更稳定的特征表示。
        if self.training and self.config.get('drop_method', None) == 'emb':
            assert self.config.get('drop_logi_k', None) is not None and self.config.get('drop_logi_m', None) is not None
            m, k = self.config.drop_logi_m, self.config.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (time_ - m)))
            p_m = p_m[:, None, None, None]	
            y_emb = y_emb.masked_fill(torch.rand_like(p_m) < p_m, 0.)


        # send to motion decoder # ... 前面是编码特征准备，此处从解码开始 ...  # 步骤1：融合编码特征、噪声特征、时间特征→[B, K, A, 128]
        emb_fusion = self.init_emb_fusion_mlp(torch.cat((encoder_out_batch, y_emb, t_emb_batch), dim=-1))	 	# [B, K, A, D]
        # 步骤2：位置编码融合→增强候选/智能体位置信息→[B, K, A, 128] emb_fusion = self.init_emb_fusion_mlp(torch.cat((y_emb, t_emb_batch), dim=-1))	 	# [B, K, A, D]
        query_token = self.post_pe_cat_mlp(emb_fusion + k_pe_batch + a_pe_batch) 								# [B, K, A, D]
        # 步骤3：Motion-Mambaformer 解码（论文核心）
        readout_token = self.motion_decoder(query_token, t_emb)													# [B, K, A, D]

        # 步骤4：生成轨迹坐标（reg_head）和置信度（cls_head）
        denoiser_x = self.reg_head(readout_token)  										# [B, K, A, 24] → 24=12帧×2维（x,y）
        denoiser_cls = self.cls_head(readout_token).squeeze(-1) 						# [B, K, A] → 候选轨迹置信度
        pred_vel = predict_vel_from_data(denoiser_x, y, time) # y = x1 t + x0 (1-t) # 步骤5：预测速度场（适配 Flow Matching 均值流）
        return [denoiser_x, denoiser_cls, pred_vel]
        # return pred_vel



class ETHIMLETransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.cfg = config

        self.objective = self.cfg.objective

        use_pre_norm = self.model_cfg.get('USE_PRE_NORM', False)

        assert not use_pre_norm, "Pre-norm is not supported in this model"

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm)

        ### serves the purpose of positional encoding
        if self.objective == 'set':
            self.motion_query_embedding = nn.Embedding(self.model_cfg.NUM_PROPOSED_QUERY, self.dim)

        self.agent_order_embedding = nn.Embedding(self.model_cfg.CONTEXT_ENCODER.AGENTS, self.dim)
        
        self.noisy_vec_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )

        self.pe_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        dim_decoder = self.model_cfg.MOTION_DECODER.D_MODEL
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(self.dim + self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(dim_decoder, dim_decoder),
            nn.ReLU(),
            nn.Linear(dim_decoder, self.model_cfg.MODEL_OUT_DIM),
        )

        self.motion_decoder = build_decoder(self.model_cfg.MOTION_DECODER, use_pre_norm, use_adaln=False)

        self.reg_head = build_mlps(c_in=self.dim, mlp_channels=self.model_cfg.REGRESSION_MLPS, ret_before_act=True, without_norm=True)

        # print out the number of parameters
        params_encoder = sum(p.numel() for p in self.context_encoder.parameters())
        params_decoder = sum(p.numel() for p in self.motion_decoder.parameters())
        params_total = sum(p.numel() for p in self.parameters())
        params_other = params_total - params_encoder - params_decoder
        logger.info("Total parameters: {:,}, Encoder: {:,}, Decoder: {:,}, Other: {:,}".format(params_total, params_encoder, params_decoder, params_other))


    def forward(self, x_data, num_to_gen=None):
        device = x_data['past_traj_original_scale'].device
        B, A, T, _ = x_data['past_traj_original_scale'].shape
        K = self.cfg.denoising_head_preds
        D = self.dim

        if self.training:
            M = self.cfg.num_to_gen
        else:
            M = num_to_gen

        # context encoder
        encoder_out = self.context_encoder(x_data['past_traj_original_scale'])  # [B, A, D]

        # init noise embeddings
        noise = torch.randn((B, M, D), device=device)       # [B, M, D]
        noise_emb = self.noisy_vec_mlp(noise)  	            # [B, M, D]


        if self.cfg.objective == 'set':
            encoder_out_batch = repeat(encoder_out, 'b a d -> b m k a d', m=M, k=K, a=A)    # [B, M, K, A, D]

            k_pe = self.motion_query_embedding(torch.arange(K, device=device))	            # [K, D]
            k_pe_batch = repeat(k_pe, 'k d -> b m k a d', b=B, m=M, a=A)	                # [B, M, K, A, D]

            a_pe = self.agent_order_embedding(torch.arange(A, device=device))               # [A, D]
            a_pe_batch = repeat(a_pe, 'a d -> b m k a d', b=B, m=M, k=K)	                # [B, M, K, A, D]

            noise_emb_batch = repeat(noise_emb, 'b m d -> b m k a d', k=K, a=A)	            # [B, M, K, A, D]
        elif self.cfg.objective == 'single':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # send to motion decoder
        emb_fusion = self.init_emb_fusion_mlp(torch.cat((encoder_out_batch, noise_emb_batch), dim=-1))	 	# [B, M, K, A, D]
        query_token = self.pe_mlp(emb_fusion + k_pe_batch + a_pe_batch) 					                # [B, M, K, A, D]

        if self.cfg.objective == 'set':
            query_token = rearrange(query_token, 'b m k a d -> (b m) k a d')
            readout_token = self.motion_decoder(query_token)
            readout_token = rearrange(readout_token, '(b m) k a d -> b m k a d', m=M)
        elif self.cfg.objective == 'single':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # readout layers
        denoiser_x = self.reg_head(readout_token)  													# [B, K, A, F * D]

        return denoiser_x

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def predict_vel_from_data(x1, xt, t):
    """

    Predict the velocity field from the predicted data.
    """
    t = pad_t_like_x(t, x1)
    # v = (x1 - xt) / (1 - t)
    v = (x1 - xt) / (0 - t)
    return v
