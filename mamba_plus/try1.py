import torch
import torch.nn as nn
import numpy as np
from layers.BiMamba4TS_layers import Encoder, EncoderLayer
from layers.Embed import PatchEmbedding, TruncateModule
from einops import rearrange
# from mamba_ssm import Mamba
# from mamba_plus import Mamba
from mamba_plus.modules.mamba_simple import Mamba
# from layers.Attention import AttentionLayer, FullAttention
from layers.BiMamba4TS_layers import EncoderLayer


encoder = Encoder(
            [
                EncoderLayer(
                    mamba_forward=Mamba(d_model=128,
                        #   batch_size=configs.batch_size,
                        #   seq_len=self.patch_num,
                          d_state=16, # 16
                          d_conv=4,  # 4
                          expand=2, # 2
                          use_fast_path=True),
                    mamba_backward=Mamba(d_model=128,
                        #   batch_size=configs.batch_size,
                        #   seq_len=self.patch_num,
                          d_state=16,
                          d_conv=4,
                          expand=2,
                          use_fast_path=True),
                    d_model=128,
                    d_ff=128,
                    dropout=0.2,
                    activation='gelu',
                    bi_dir=1,
                    residual=1
                ) for _ in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(128)
        ).to("cuda")

batch, length, dim = 2, 10, 128
x = torch.randn(batch, length, dim).to("cuda")

output = encoder(x)
print("output:",output.shape)