import numpy as np
from scipy.stats import gaussian_kde
from einops import rearrange

def compute_kde_nll(predicted_trajs, gt_traj):
    """
    计算预测轨迹和真实轨迹之间的 KDE 负对数似然。

    Args:
        predicted_trajs (np.ndarray): 预测轨迹，形状为 (num_batches, num_samples, num_timesteps, 2)
        gt_traj (np.ndarray): 真实轨迹，形状为 (num_timesteps, num_samples, 2)

    Returns:
        float: KDE 负对数似然
    """
    gt_traj = gt_traj.permute(1, 0, 2).repeat(1,20,1)  # (12,B,2)
    gt_traj = gt_traj.cpu().detach().numpy()
    predicted_trajs = predicted_trajs.cpu().detach().numpy()

    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                # 提取当前批次和时间步的预测轨迹
                current_predicted_trajs = predicted_trajs[batch_num, :, timestep, :].T
                # 计算 KDE
                kde = gaussian_kde(current_predicted_trajs)
                # 计算真实轨迹点的对数概率密度
                pdf = np.clip(kde.logpdf(gt_traj[timestep, :, :].T), a_min=log_pdf_lower_bound, a_max=None)
                # 累加对数似然
                kde_ll += np.mean(pdf) / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                # 如果出现线性代数错误，返回 NaN
                kde_ll = np.nan
                break
        if np.isnan(kde_ll):
            break

    return -kde_ll if not np.isnan(kde_ll) else np.nan

# # 示例数据
# num_batches = 2  # 批次数
# num_samples = 100  # 每个批次的样本数
# num_timesteps = 12  # 时间步数
#
# # 生成预测轨迹数据 (num_batches, num_samples, num_timesteps, 2)
# predicted_trajs = np.random.randn(num_batches, num_samples, num_timesteps, 2)
#
# # 生成真实轨迹数据 (num_timesteps, num_samples, 2)
# gt_traj = np.random.randn(num_timesteps, num_samples, 2)
#
# # 计算 KDE 负对数似然
# kde_nll = compute_kde_nll(predicted_trajs, gt_traj) # 预测轨迹：(B,K,12,2)   真实轨迹：(12,K,2)
#
# print(f"KDE 负对数似然: {kde_nll}")