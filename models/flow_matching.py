import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

from collections import namedtuple

from einops import rearrange, reduce, repeat
from functools import partial

from tqdm.auto import tqdm
from utils.normalization import unnormalize_min_max, unnormalize_sqrt
from utils.utils import apply_mask
from utils.utils import LossBuffer

ModelPrediction = namedtuple('ModelPrediction', ['pred_vel', 'pred_data', 'pred_score'])


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def stopgrad(x):                # 对张量 x 执行 detach()，停止梯度传播。
    return x.detach()

def extract(a, t, x_shape):         #从张量 a 中提取索引为 t 的元素，并调整形状以匹配 x 的维度（方便广播）。
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FlowMatcher(nn.Module):
    def __init__(
        self,
        cfg,
        model,
        logger
    ):
        super().__init__()

        # init
        self.cfg = cfg
        self.model = model
        self.logger = logger

        self.num_agents = cfg.agents
        self.out_dim = cfg.MODEL.MODEL_OUT_DIM

        self.objective = cfg.objective
        self.sampling_steps = cfg.sampling_steps
        self.solver = cfg.get('solver', 'euler')

        assert cfg.objective in {'pred_vel', 'pred_data'}, 'objective must be either pred_vel or pred_data'
        assert self.cfg.get('LOSS_VELOCITY', False) == False, 'Velocity loss is not supported yet.'

        # special normalization params
        if self.cfg.get('data_norm', None) == 'sqrt':
            self.sqrt_a_ = torch.tensor([self.cfg.sqrt_x_a, self.cfg.sqrt_y_a], device=self.device)
            self.sqrt_b_ = torch.tensor([self.cfg.sqrt_x_b, self.cfg.sqrt_y_b], device=self.device)

        # set up the loss buffer
        self.loss_buffer = LossBuffer(t_min=0, t_max=1.0, num_time_steps=100)

        # register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        

    @property
    def device(self):
        return self.cfg.device
    
    def get_precond_coef(self, t):
        """
        Get preconditioned wrapper coefficients.
        D_theta = alpha_t * x_t + beta_t * F_theta
        @param t: [B]
        """
        coef_1 = t.pow(2) * self.cfg.sigma_data ** 2 + (1-t).pow(2)
        alpha_t = t * self.cfg.sigma_data ** 2 / coef_1
        beta_t = (1 - t) * self.cfg.sigma_data / coef_1.sqrt()

        return alpha_t, beta_t
    
    def get_input_scaling(self, t):
        """
        Get the input scaling factor.
        """
        var_x_t = self.cfg.sigma_data ** 2 * t.pow(2) + (1 - t).pow(2)
        return 1.0 / var_x_t.sqrt().clip(min=1e-4, max=1e4)

    def fm_wrapper_func(self, x_t, t, model_out):           #对模型输出 model_out 进行包装，使其匹配流匹配的目标
        """
        Build wrapper for network regression output. We don't modify the classification logits.
        We aim to let the wrapper to match the data prediction (x_1 in the flow model).
        @param x_t: 		[B, K, A, F * D]
        @param t: 			[B]
        @param model_out: 	[B, K, A, F * D]
        """
        if self.cfg.fm_wrapper == 'direct':  #不做任何修改，直接返回 model_out
            return model_out
        elif self.cfg.fm_wrapper == 'velocity':       #通过 x_t + (1 - t) * model_out 计算，将 model_out 视为 “速度项”，结
            # 合当前 x_t 和时间步 t 生成接近 x_1 的预测，(1 - t) 确保随时间步推进（t 接近 1），model_out 权重降低
            t = pad_t_like_x(t, x_t)
            return x_t + (1 - t) * model_out
        elif self.cfg.fm_wrapper == 'precond':
            t = pad_t_like_x(t, x_t)
            alpha_t, beta_t = self.get_precond_coef(t)
            return alpha_t * x_t + beta_t * model_out


    def predict_vel_from_data(self, x1, xt, t):
        """
        Predict the velocity field from the predicted data.
        """
        t = pad_t_like_x(t, x1)
        # v = (x1 - xt) / (1 - t)
        v = (x1 - xt) / (0 - t)
        return v

    def predict_data_from_vel(self, v, xt, t):
        """
        Predict the data from the predicted velocity field.
        """
        t = pad_t_like_x(t, xt)
        x1 = xt + v * (1 - t)
        return x1

    def fwd_sample_t(self, x0, x1, t):
        """
        Sample the latent space at time t.
        """
        t = pad_t_like_x(t, x0)
        # xt = t * x1 + (1 - t) * x0      # simple linear interpolation
        # ut = x1 - x0                    # xt derivative w.r.t. t
        xt = t * x0 + (1 - t) * x1 # t=0为真实值　t=1为噪声
        ut = x0 - x1
        return xt, ut

    # 损失权重控制器，通过不同的策略（直接、速度加权、预处理加权）调整不同时间步t对损失的贡献
    def get_reweighting(self, t, wrapper=None):
        wrapper = default(wrapper, self.cfg.fm_wrapper)
        if wrapper == 'direct':
            l_weight = torch.ones_like(t)
        elif wrapper == 'velocity':
            l_weight = 1.0 / (1 - t) ** 2
        elif wrapper == 'precond':
            alpha_t, beta_t = self.get_precond_coef(t)
            l_weight = 1.0 / beta_t ** 2
        if self.cfg.fm_rew_sqrt:
            l_weight = l_weight.sqrt()
        l_weight = l_weight.clamp(min=1e-4, max=1e4)
        return l_weight
    
    def get_loss_input(self, y_start_k):
        """
        Prepare the input for the flow matching model training.
        """

        # random time steps to inject noise
        bs = y_start_k.shape[0]          # 批次大小
        if self.cfg.t_schedule == 'uniform':
            t = torch.rand((bs, ), device=self.device)
        elif self.cfg.t_schedule == 'logit_normal':
            # note: this is logit-normal (not log-normal)
            mean_ = self.cfg.logit_norm_mean
            std_ = self.cfg.logit_norm_std
            t_normal_ = torch.randn((bs, ), device=self.device) * std_ + mean_
            t = torch.sigmoid(t_normal_)
            r_normal_ = torch.randn((bs, ), device=self.device) * std_ + mean_
            r = torch.sigmoid(r_normal_)
        else:
            if '==' in self.cfg.t_schedule:
                # constant_t
                t = float(self.cfg.t_schedule.split('==')[1]) * torch.ones((bs, ), device=self.device)
            else:
                # custom two-stage uniform distribution
                # e.g., 't0.5_p0.3' means with 30% probability, sample from [0, 0.5] uniformly, and with 70% probability, sample from [0.5, 1] uniformly
                # 自定义两步分布（如 30% 概率采样 [0,0.5]，70% 采样 [0.5,1]）
                cutoff_t = float(self.cfg.t_schedule.split('_')[0][1:])
                prob_1 = float(self.cfg.t_schedule.split('_')[1][1:])

                t_1 = torch.rand((bs, ), device=self.device) * cutoff_t
                t_2 = cutoff_t + torch.rand((bs, ), device=self.device) * (1 - cutoff_t)
                rand_num = torch.rand((bs, ), device=self.device)

                t = t_1 * (rand_num < prob_1) + t_2 * (rand_num >= prob_1)



        assert t.min() >= 0 and t.max() <= 1
        assert r.min() >= 0 and r.max() <= 1

        # noise sample
        if self.cfg.tied_noise:
            # 所有预测分支共享同一噪声
            noise = torch.randn_like(y_start_k[:, 0:1])                                  # [B, 1, T, D]
            noise = noise.expand(-1, self.cfg.denoising_head_preds, -1, -1)              # [B, K, T, D]
        else:
            # 每个分支独立噪声
            noise = torch.randn_like(y_start_k)                                          # [B, K, T, D]

        # sample the latent space at time t
        # 生成时间 t 的带噪声样本 xt 和理论速度 ut
        x_t, u_t = self.fwd_sample_t(x0=noise, x1=y_start_k, t=t)                        # [B, K, T, D] * 2

        if self.objective == 'pred_data':
            target = y_start_k
        elif self.objective == 'pred_vel':
            target = u_t
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 损失权重（根据时间步t调整）
        l_weight = self.get_reweighting(t)

        return t, r, x_t, u_t, target, l_weight

        # 模型预测与推理
    def model_predictions(self, y_t, x, t, flag_print):
        if self.cfg.fm_in_scaling:
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t)
        else:
            y_t_in = y_t
        '''
        # 2. 【注意：这是调试用的临时代码，正常训练/评估需删除！】
        作用：将模型输入y_t_in替换为纯高斯噪声，而非原始带噪轨迹
        问题：会覆盖真实输入，导致模型无法利用当前带噪轨迹的信息，仅用于测试模型是否能从纯噪声生成轨迹
        '''
        # 假设 y_t_in 是一个张量
        y_t_in = torch.randn_like(y_t_in)  # 生成与 y_t_in 形状相同的随机张量

        t = torch.ones((y_t_in.shape[0]),device="cuda")    # 全1
        r = torch.zeros((y_t_in.shape[0]),device="cuda")    # 全0
        # 4. 调用核心模型（ETHMotionTransformer），输入带噪轨迹和上下文，得到预测结果
        model_out, pred_score, pred_vel = self.model(y_t_in, t, r, x_data=x)
        # 5. 包装模型输出：将model_out转换为符合Flow Matching目标的“去噪后轨迹数据”y_data_at_t
        # fm_wrapper_func：根据配置的wrapper模式（如direct/velocity/precond）处理model_out
        # 示例（direct模式）：直接返回model_out作为y_data_at_t（即模型预测的去噪后轨迹）
        y_data_at_t = self.fm_wrapper_func(y_t, t, model_out)            # [B, K, A, F * D]

        if self.objective == 'pred_vel':
            raise NotImplementedError  # 直接预测速度场的分支（当前未实现，因你的目标是pred_data）

        elif self.objective == 'pred_data':
            gt_y_data = rearrange(x['fut_traj'], 'b a f d -> b 1 a (f d)')

            this_t = round(t.unique().item(), 4)

            if flag_print:  # 3. 若需要打印日志（flag_print=True），计算近似minADE并打印时间步
                y_data_ = rearrange(y_data_at_t, 'b k a (f d) -> (b a) k f d', f=self.cfg.future_frames)
                # 输出gt_y_data形状：[B, 1, A, F*D]（K=1，因真实轨迹只有1条，匹配预测的[B,K,A,F*D]）
                gt_y_data = rearrange(gt_y_data, 'b k a (f d) -> (b a) k f d', f=self.cfg.future_frames)

                if self.cfg.get('data_norm', None) == 'min_max':
                    y_data_metric = unnormalize_min_max(y_data_, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
                    gt_y_data_metric = unnormalize_min_max(gt_y_data, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
                elif self.cfg.get('data_norm', None) == 'sqrt':
                    y_data_metric = unnormalize_sqrt(y_data_, self.sqrt_a_, self.sqrt_b_)
                    gt_y_data_metric = unnormalize_sqrt(gt_y_data, self.sqrt_a_, self.sqrt_b_)
                elif self.cfg.get('data_norm', None) == 'original':
                    y_data_metric = y_data_
                    gt_y_data_metric = gt_y_data

                # 3.3 计算近似minADE（快速评估当前去噪效果）
                error_metric = (y_data_metric - gt_y_data_metric).abs()  # [B * A, K, F, D]
                batch_min_ade_approx = error_metric.norm(dim=-1, p=2).mean(dim=-1).min(dim=-1).values.mean()
                if this_t == 0.0:
                    self.logger.info("{}".format("-" * 50))
                # self.logger.info("Sampling time step: {:.3f}, batch minADE approx: {:.4f}".format(this_t, batch_min_ade_approx))
                self.logger.info("Sampling time step: {:.3f}".format(this_t))
            # 4. 反推速度场：从预测的轨迹数据y_data_at_t反推Flow Matching所需的速度场
            # 调用predict_vel_from_data，公式：v=(x1 - xt)/(0 - t)
            # 其中x1=y_data_at_t（预测的去噪后轨迹），xt=y_t（当前带噪轨迹），t=当前时间步
            # 分母用“0 - t”而非“1 - t”：因你的时间步定义可能调整（t=1→纯噪声，t=0→真实轨迹）
            pred_vel = self.predict_vel_from_data(y_data_at_t, y_t, t)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_vel, y_data_at_t, pred_score)

    @torch.inference_mode()
    def bwd_sample_t(self, y_t: torch.tensor, t: int, dt: float, x_data: dict, flag_print: bool=False):
        """反向采样单步：从当前样本 y_t 沿流场更新到下一步 y_next"""

        B, K, T, D = y_t.shape

        batched_t = torch.full((B,), t, device=self.device, dtype=torch.float)
        model_preds = self.model_predictions(y_t, x_data, batched_t, flag_print)

        # y_next = y_t + model_preds.pred_vel * dt
        # y_next = y_t - model_preds.pred_vel * dt
        y_next = y_t - model_preds.pred_vel
        return y_next, model_preds.pred_data, model_preds

    @torch.no_grad()
    def sample(self, x_data, num_trajs, return_all_states=False):
        """
        Sample from the model.完整采样流程：从噪声出发，通过多步反向采样生成目标数据
        """
        # start with y_T ~ N(0,I), reversed MC to conditionally denoise the traj
        # 1. 断言：确保生成的轨迹数（num_trajs）与配置一致（避免维度错配）
        # num_trajs：外部传入的需生成的轨迹数（如评估时的K=20）
        # self.cfg.denoising_head_preds：配置中预设的候选轨迹数（K=20）
        assert num_trajs == self.cfg.denoising_head_preds, 'num_trajs must be equal to denoising_head_preds = {}'.format(self.cfg.denoising_head_preds)
        y_data = None

        batch_size = x_data['batch_size']

        # 3. 生成初始高斯噪声y_t（Flow Matching反向过程的起点）
        # 形状：[B, K, A, D] → B=批次，K=20候选，A=2智能体，D=24（12帧×2维坐标）
        y_t = torch.randn((batch_size, num_trajs, self.num_agents, self.out_dim), device=self.device)
        if self.cfg.tied_noise: # 所有分支共享同一噪声
            y_t = y_t[:, :1].expand(-1, self.cfg.denoising_head_preds, -1, -1)

        # sampling loop
        y_data_at_t_ls = []
        t_ls = []
        y_t_ls = []

        # 简化的时间步设置（完整代码中支持多种solver，此处为单步示例）欧拉法 均匀时间步（适合简单场景，每个步骤去噪强度一致）
        dt = 1.0 / self.sampling_steps    # 步长=1/总采样步数（如sampling_steps=10→dt=0.1）
        t_ls = dt * np.arange(self.sampling_steps)  # 时间步序列：[0.1, 0.2, ..., 1.0]（t从0.1到1.0，共10步）
        t_ls = t_ls[::-1] + 0.1
        dt_ls = dt * np.ones(self.sampling_steps) # 所有步长相同：[0.1, 0.1, ..., 0.1]
        # if self.solver == 'euler':
        #     dt = 1.0 / self.sampling_steps
        #     t_ls = dt * np.arange(self.sampling_steps)
        #     dt_ls = dt * np.ones(self.sampling_steps)
        # elif self.solver == 'lin_poly':
        #     # linear time growth in the first half with small dt
        #     # polinomial growth of dt in the second half
        #     lin_poly_long_step = self.cfg.lin_poly_long_step
        #     lin_poly_p = self.cfg.lin_poly_p
        #
        #     n_steps_lin = self.sampling_steps // 2
        #     n_steps_poly = self.sampling_steps - n_steps_lin
        #
        #     dt_lin = 1.0 / lin_poly_long_step
        #     t_lin_ls = dt_lin * np.arange(n_steps_lin)
        #
        #     def _polynomially_spaced_points(a, b, N, p=2):
        #         # Generate N points in the interval [a, b] with spacing determined by the power p.
        #         points = [a + (b - a) * ((i - 1) ** p) / ((N - 1) ** p) for i in range(1, N + 1)]
        #         return points
        #
        #     t_poly_start = t_lin_ls[-1] + dt_lin
        #     t_poly_end = 1.0
        #     t_poly_ls_ = _polynomially_spaced_points(t_poly_start, t_poly_end, n_steps_poly + 1, p=lin_poly_p)
        #     dt_poly = np.diff(t_poly_ls_)
        #
        #     dt_ls = np.concatenate([dt_lin * np.ones(n_steps_lin), dt_poly]).tolist()
        #     t_ls = np.concatenate([t_lin_ls, t_poly_ls_[:-1]]).tolist()
        #
        # else:
        #     raise NotImplementedError(f"Unknown solver: {self.solver}")

        # define the time steps to print
        num_prints = 10
        # if len(t_ls) > num_prints:
        #     print_times = t_ls[::self.sampling_steps // num_prints]
        #     if t_ls[-1] not in print_times:
        #         print_times.append(t_ls[-1])
        # else:
        #     print_times = t_ls

        # for idx_step, (cur_t, cur_dt) in enumerate(zip(t_ls, dt_ls)):
        #     flag_print = cur_t in print_times
        #     y_t, y_data, model_preds = self.bwd_sample_t(y_t, cur_t, cur_dt, x_data, flag_print)
        #     y_data_at_t_ls.append(y_data)
        #     if return_all_states:
        #         y_t_ls.append(y_t)

        # 单步反向去噪：直接调用bwd_sample_t，cur_t=1（纯噪声阶段），cur_dt=1（一步完成所有去噪）
        y_t, y_data, model_preds = self.bwd_sample_t(y_t, 1, 1, x_data, flag_print=True)
        # 保存单步的中间预测轨迹
        y_data_at_t_ls.append(y_data)
        # 保存单步的去噪后状态（若需）
        if return_all_states:
            y_t_ls.append(y_t)

        # 整理输出
        # 1. 整理中间预测轨迹：堆叠为[B, S, K, A, F*D]（B=批次，S=采样步数，K=候选，A=智能体，F*D=未来帧×坐标）
        y_data_at_t_ls = torch.stack(y_data_at_t_ls, dim=1)     # [B, S, K, A, F * D]
        t_ls = torch.tensor(t_ls, device=self.device)   # [S]
        if return_all_states:
            y_t_ls = torch.stack(y_t_ls, dim=1)  # [B, S, K, A, F * D]

        # 4. 返回结果：最终去噪轨迹、中间预测轨迹、时间步、中间状态、候选置信度
        return y_t, y_data_at_t_ls, t_ls, y_t_ls, model_preds.pred_score

    def p_losses(self, x_data, log_dict=None):
        """
        Denoising model training. 计算训练损失：包括速度场误差和数据误差
        """

        # init
        B, A = x_data['fut_traj'].shape[:2]
        # print(A)
        K = self.cfg.denoising_head_preds  # 预测分支数
        T = self.cfg.future_frames  # 未来帧数量
        assert self.objective == 'pred_data', 'only pred_data is supported for now'

        # forward process to create noisy samples
        fut_traj_normalized = repeat(x_data['fut_traj'], 'b a f d -> b k a (f d)', k=K)
        t, r, y_t, u_t, _, l_weight = self.get_loss_input(y_start_k = fut_traj_normalized)
        #######
        num_selected = int(0.5 * B)
        indices = np.random.permutation(B)[:num_selected]
        r[indices] = t[indices]

        # model pass
        if self.cfg.fm_in_scaling:
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t)
        else:
            y_t_in = y_t
        # 模型输入预处理（缩放）
        if self.training and self.cfg.get('drop_method', None) == 'input':
            assert self.cfg.get('drop_logi_k', None) is not None and self.cfg.get('drop_logi_m', None) is not None
            m, k = self.cfg.drop_logi_m, self.cfg.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (t - m)))
            p_m = p_m[:, None, None, None]
            y_t_in = y_t_in.masked_fill(torch.rand_like(p_m) < p_m, 0.)

        # 模型前向传播（输出去噪结果、分类logits、速度预测）
        model_out, denoiser_cls, pred_vel_ = self.model(y_t_in, t, r, x_data=x_data)  # [B, K, A, T * D] + [B, K, A] 进入到backbone_eth_ucy的forward
        denoised_y = self.fm_wrapper_func(y_t, t, model_out) # 应用流匹配包装器（调整输出格式）

        # from thop import profile # 计算复杂度，模型参数量
        # # 假设 model 是你的模型实例，input 是模型的输入
        # # input = torch.randn(1, 20, 12, 2)  # 示例输入
        # macs, params = profile(self.model, inputs=(y_t_in, t, r, x_data))
        #
        # print(f"计算量 (MACs): {macs}")
        # print(f"参数量 (Params): {params}")

        # model_partial = partial(self.model, x_data=x_data)
        # 定义一个函数，只返回self.model的最后一个输出
        def pred_vel(y_t, t, r):
            return self.model(y_t, t, r, x_data=x_data)[-1]

        # 计算雅可比向量积（JVP）：用于求解速度场对时间的导数
        v, dvdt = torch.autograd.functional.jvp( # v:(B,20,1,24) dvdt:(B,20,1,24)
            # lambda y_t, t, r: model_partial(y_t, t, r),
            # lambda y_t, t, r: self.model(y_t_in, t, r, x_data=x_data),
            pred_vel, #
            (y_t, t, r),
            (u_t, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )
        t = pad_t_like_x(t, u_t)
        r = pad_t_like_x(r, u_t)
        v_tgt = u_t - (t - r) * dvdt # (B,20,1,24)         # 目标速度场
        error = v - stopgrad(v_tgt) # (B,20,1,24)          # 速度场误差
        t = t.squeeze(3).squeeze(2).squeeze(1)

        # component selection
        denoised_y = rearrange(denoised_y, 'b k a (f d) -> b k a f d', f = self.cfg.future_frames)
        fut_traj_normalized = fut_traj_normalized.view(B, K, A, T, 2)

        # 反归一化：将数据从归一化空间转换回原始度量空间（如像素坐标、物理坐标）
        #训练时为了让优化更稳定（避免数据量级差异导致梯度爆炸），会将轨迹坐标归一化到[-1, 1]（min_max）或其他范围，
        # 但计算损失时必须用真实物理坐标（如 “米”），否则误差无实际意义
        if self.cfg.get('data_norm', None) == 'min_max':
            denoised_y_metric = unnormalize_min_max(denoised_y, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1) 		 # [B, K, A, T, D]
            fut_traj_metric = unnormalize_min_max(fut_traj_normalized, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)  # [B, K, A, T, D]
        elif self.cfg.get('data_norm', None) == 'sqrt':
            denoised_y_metric = unnormalize_sqrt(denoised_y, self.sqrt_a_, self.sqrt_b_)            # [B, K, A, T, D]
            fut_traj_metric = unnormalize_sqrt(fut_traj_normalized, self.sqrt_a_, self.sqrt_b_)     # [B, K, A, T, D]
        elif self.cfg.get('data_norm', None) == 'original':
            denoised_y_metric = denoised_y
            fut_traj_metric = fut_traj_normalized
        else:
            raise ValueError(f"Unknown data normalization method: {self.cfg.get('data_norm', None)}")

        if self.cfg.get('LOSS_VELOCITY', False):
            raise NotImplementedError
            denoised_y_metric = rearrange(denoised_y_metric, 'b k a (f d) -> b k a f d', f = self.cfg.future_frames, d = 4)
            denoised_y_metric_xy, denoised_y_metric_v = denoised_y_metric[..., :2], denoised_y_metric[..., 2:4]

            gt_traj_vel = x_data['fut_traj_vel'][:, None].expand(-1, K, -1, -1, -1)  # [B, K, A, T, 2]
            loss_reg_vel = F.l1_loss(denoised_y_metric_v, gt_traj_vel, reduction='none').mean()
        else:
            denoised_y_metric_xy = denoised_y_metric
            loss_reg_vel = torch.zeros(1).to(self.device)

        # 计算去噪轨迹与真实轨迹的L2误差（按代理和时间步） # 计算回归损失（预测轨迹与真实轨迹的L1/L2误差）
        denoising_error_per_agent = (denoised_y_metric_xy - fut_traj_metric).view(B, K, A, T, 2).norm(dim=-1)  	 # [B, K, A, T] # (B,20,1,12)

        if self.cfg.get('LOSS_REG_SQUARED', False):
            denoising_error_per_agent = denoising_error_per_agent ** 2

        denoising_error_per_scene = denoising_error_per_agent.mean(dim=-2)  # [B, K, T] # (B,20,12)

        if self.cfg.get('LOSS_REG_REDUCTION', 'mean') == 'mean':
            denoising_error_per_scene = denoising_error_per_scene.mean(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.mean(dim=-1)
        elif self.cfg.get('LOSS_REG_REDUCTION', 'mean') == 'sum':
            denoising_error_per_scene = denoising_error_per_scene.sum(dim=-1) # (B,20,)
            denoising_error_per_agent = denoising_error_per_agent.sum(dim=-1) # (B,20,1)
        else:
            raise ValueError(f"Unknown reduction method: {self.cfg.get('LOSS_REG_REDUCTION', 'mean')}")

        if self.cfg.LOSS_NN_MODE == 'scene':
            # scene-level selection 场景级选择：每个场景选最优分支（误差最小的K分支）
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B] 选误差最小的分支
            # 2. 取出最优候选的误差：gather按索引取数，squeeze去掉多余维度
            loss_reg_b = denoising_error_per_scene.gather(1, selected_componnts[:, None]).squeeze(1)  		# [B]
            # 分类损失：分类器预测最优分支
            cls_logits = denoiser_cls.mean(dim=-1)  # [B, K] # [B, K] → 按智能体平均置信度（A=1，即原置信度）
            # [B] → 交叉熵损失
            loss_cls_b = F.cross_entropy(input=cls_logits, target=selected_components, reduction='none')	# [B]
        elif self.cfg.LOSS_NN_MODE == 'agent':
            # agent-level selection 以 “每个智能体” 为单位选最优候选（适合多智能体场景，比如 2 个行人各选最优候选）。
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A] # (B,1) 20条轨迹最小值索引
            loss_reg_b = denoising_error_per_agent.gather(1, selected_components[:, None, :]).squeeze(1)  	# [B, A] # (128,1)
            loss_reg_b = loss_reg_b.mean(dim=-1)  # [B] # (128)
            ###### # MeanFlow-loss  速度场误差（来自之前JVP计算的error）
            error = error.squeeze(2).sum(dim=-1).unsqueeze(2) # (B,20,1)
            # # import ipdb
            # # ipdb.set_trace()
            loss_error = error.gather(1, selected_components[:, None, :]).squeeze(1)  # (B,1)
            loss_error = adaptive_l2_loss(loss_error) # (1,)
            ######
            cls_logits = rearrange(denoiser_cls, 'b k a -> (b a) k')	# [B * A, K]
            cls_labels = selected_components.view(-1)					# [B * A]
            loss_cls_b = F.cross_entropy(input=cls_logits, target=cls_labels, reduction='none')	 # [B * A]
            loss_cls_b = loss_cls_b.view(B, A).mean(dim=-1)  	# [B]
        elif self.cfg.LOSS_NN_MODE == 'both':
            # scene-level selection 结合场景级和智能体级的损失，
            # 通过 omega 权重平衡两者贡献（比如 omega=0.5 表示各占一半），分类损失暂设为 0（预留扩展空间）。
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B]
            loss_reg_b_scene = denoising_error_per_scene.gather(1, selected_components[:, None]).squeeze(1)  		# [B] 

            # agent-level selection
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A]
            loss_reg_b = denoising_error_per_agent.gather(1, selected_components[:, None, :]).squeeze(1)  	# [B, A]
            loss_reg_b_agent = loss_reg_b.mean(dim=-1)  # [B]
            # 3. 混合损失：按权重omega加权（scene损失 + agent损失）
            loss_reg_b = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('omega', 1.0)  * loss_reg_b_scene + loss_reg_b_agent

            # dummy input for loss_cls_b  # 4. 分类损失占位（实际需根据需求实现，此处设为0）
            loss_cls_b = torch.zeros_like(loss_reg_b)


        # loss computation# 1. 计算加权回归损失：乘以时间步权重（l_weight来自get_reweighting，调整不同t的损失贡献）
        loss_reg = (loss_reg_b * l_weight).mean()  # scalar

        loss_cls = loss_cls_b.mean() # 2. 分类损失：批次平均

        loss_error = loss_error.mean()  # 3. 速度场误差损失：批次平均
        # loss_error = torch.tensor([0.0])

        # 4. 读取损失权重配置（从OPTIMIZATION.LOSS_WEIGHTS）
        weight_reg = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('reg', 1.0)
        weight_cls = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('cls', 1.0)
        weight_vel = self.cfg.OPTIMIZATION.LOSS_WEIGHTS.get('vel', 0.2)

        # loss = weight_reg * loss_reg.mean() + weight_cls * loss_cls.mean() + weight_vel * loss_reg_vel.mean()
        # 总损失 = 回归损失*权重 + 分类损失*权重 + 速度场误差损失*权重
        loss = weight_reg * loss_reg.mean() + weight_cls * loss_cls.mean() + loss_error.mean() * 5
        # loss = weight_reg * loss_reg.mean() + weight_cls * loss_cls.mean()

        # record the loss for each denoising level
        # 6. 记录每个时间步t的损失（用于分析不同去噪阶段的损失变化
        flag_reset = self.loss_buffer.record_loss(t, loss_reg_b.detach(), epoch_id=log_dict['cur_epoch'])
        if flag_reset:
            dict_loss_per_level = self.loss_buffer.get_average_loss()
            log_dict.update({
                'denoiser_loss_per_level': dict_loss_per_level
            })

        return loss, loss_reg.mean(), loss_cls.mean(), loss_error.mean()


    def forward(self, x, log_dict=None):
        return self.p_losses(x, log_dict)

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (batch, dim)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.sum(error ** 2, dim=-1)  # ||Δ||^2 per sample # (B,3,32,32)-->(B,3,32)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2 # # (B,3,32)
    return (stopgrad(w) * loss).mean()
