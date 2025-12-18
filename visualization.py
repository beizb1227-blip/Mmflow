import torch
import matplotlib.pyplot as plt
import numpy as np

#### 可视化 # 一条条的绘制
def draw_one(past_traj, fut_traj, tra_pred, traj_scale, count):

    print(f"past_traj:{past_traj.shape}\nfut_traj:{fut_traj.shape}\nx_t:{tra_pred.shape}")
    # past_traj: torch.Size([11, 10, 6])
    # fut_traj: torch.Size([11, 20, 20, 2])
    # x_t: torch.Size([11, 20, 20, 2])
    past_traj = past_traj[:, :, 2:4]
    num = fut_traj.size(0)
    for j in range(num):  # epoch
        print(f"i:{j}")
        # 提取每个张量第一维的数据用于可视化，这里只取了第一维（索引为0）的数据，你可以根据需要调整
        past_traj_1st = past_traj[j].cpu().numpy() * traj_scale  # 移动到CPU并转换为numpy数组，方便后续处理
        fut_traj_1st = fut_traj[j].cpu().numpy() * traj_scale
        x_t_1st = tra_pred[j].cpu().numpy() * traj_scale
        # 设置不同的颜色用于区分不同的张量轨迹
        past_color = 'g'  # 绿色表示past_traj
        fut_color = 'b'  # 蓝色表示fut_traj
        x_t_color = 'r'  # 红色表示x_t
        # 循环绘制每个索引对应的fut_traj和x_t的轨迹图
        for i in range(fut_traj_1st.shape[0]):
            fig, ax = plt.subplots()  # 创建新的图像和坐标轴对象

            # 绘制past_traj的轨迹（假设past_traj的坐标信息在最后两维，这里根据实际情况调整索引）
            ax.plot(past_traj_1st[:, -2], past_traj_1st[:, -1], color=past_color,
                    label='Past Trajectory')

            # 标记过去轨迹的终点
            x_end_past, y_end_past = past_traj_1st[-1, -2], past_traj_1st[-1, -1]
            ax.plot(x_end_past, y_end_past, color=past_color, marker='*', markersize=8,
                    label='Past Trajectory End')

            # 绘制当前索引下fut_traj的轨迹，并标记终点
            x_end_fut, y_end_fut = fut_traj_1st[i, -1, -2], fut_traj_1st[i, -1, -1]
            ax.plot(x_end_fut, y_end_fut, color=fut_color, marker='*', markersize=8,
                    label='Future Trajectory End')
            ax.plot(fut_traj_1st[i, :, -2], fut_traj_1st[i, :, -1], color=fut_color)

            # 绘制当前索引下x_t的轨迹，并标记终点
            x_end_xt, y_end_xt = x_t_1st[i, -1, -2], x_t_1st[i, -1, -1]
            ax.plot(x_end_xt, y_end_xt, color=x_t_color, marker='*', markersize=8,
                    label='Predict Trajectory End')
            ax.plot(x_t_1st[i, :, -2], x_t_1st[i, :, -1], color=x_t_color)

            # 用蓝色线条连接观测轨迹的终点和未来轨迹的起点
            fut_start_x, fut_start_y = fut_traj_1st[i, 0, -2], fut_traj_1st[i, 0, -1]
            ax.plot([x_end_past, fut_start_x], [y_end_past, fut_start_y], color='b', linestyle='--')

            # 用红色线条连接观测轨迹的终点和预测轨迹的起点
            xt_start_x, xt_start_y = x_t_1st[i, 0, -2], x_t_1st[i, 0, -1]
            ax.plot([x_end_past, xt_start_x], [y_end_past, xt_start_y], color='r', linestyle='--')

            # 添加标题、坐标轴标签和图例等，使图像更清晰
            ax.set_title(f'Trajectory Visualization - Index {i}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.legend()

            # 显示当前图像
            # plt.show()
            # 保存当前图像到./image文件夹下
            plt.savefig(f'./results/led_augment/try3-26-sdd-leepfrog/img-1/trajectory_{count}_{j}_{i}.png')
            plt.close()


#### 可视化 ## 20条一起绘制
def draw_more(past_traj, fut_traj, tra_pred, traj_scale, count):
    print(f"past_traj:{past_traj.shape}\nfut_traj:{fut_traj.shape}\nx_t:{tra_pred.shape}")
    past_traj = past_traj[:, :, 2:4]  # 提取坐标信息
    num = fut_traj.size(0)

    for j in range(num):
        # 提取并缩放轨迹数据
        past_traj_1st = past_traj[j, :8].cpu().numpy() * traj_scale  # 仅取前8帧作为观测轨迹
        fut_traj_1st = fut_traj[j].cpu().numpy() * traj_scale
        x_t_1st = tra_pred[j].cpu().numpy() * traj_scale

        # 创建画布
        fig, ax = plt.subplots()

        # 绘制观测轨迹（绿色实线+绿色圆点）
        ax.plot(past_traj_1st[:, 0], past_traj_1st[:, 1],
                color='g', linestyle='-', linewidth=0.8,
                marker='o', markersize=2, markerfacecolor='g')

        # 标记观测轨迹终点（蓝色星号）
        x_end_past, y_end_past = past_traj_1st[-1, 0], past_traj_1st[-1, 1]
        ax.plot(x_end_past, y_end_past,
                color='b', marker='o', markersize=2, linestyle='none')

        for i in range(fut_traj_1st.shape[0]):
            # 真实未来轨迹处理
            fut_start_x, fut_start_y = fut_traj_1st[i, 0, 0], fut_traj_1st[i, 0, 1]
            # 连接线（蓝色实线）
            ax.plot([x_end_past, fut_start_x], [y_end_past, fut_start_y],
                    color='b', linestyle='-', linewidth=0.8)
            # 真实轨迹（蓝色实线）
            ax.plot(fut_traj_1st[i, :, 0], fut_traj_1st[i, :, 1],
                    color='b', linestyle='-', linewidth=0.8,
                    marker='o', markersize=2, markerfacecolor='b')

            # 预测轨迹处理
            xt_start_x, xt_start_y = x_t_1st[i, 0, 0], x_t_1st[i, 0, 1]
            # 连接线（红色虚线）
            ax.plot([x_end_past, xt_start_x], [y_end_past, xt_start_y],
                    color='r', linestyle='--', linewidth=0.8)
            # 预测轨迹（红色虚线）
            ax.plot(x_t_1st[i, :, 0], x_t_1st[i, :, 1],
                    color='r', linestyle='--', linewidth=0.8)

            # 预测终点（红色五角星）
            ax.plot(x_t_1st[i, -1, 0], x_t_1st[i, -1, 1],
                    color='r', marker='*', markersize=5, linestyle='none')

        # 保存当前图像
        plt.savefig(f'results_sdd/cor_fm/7-6/img20/trajectory_{count}_{j}.png')
        plt.close()

# def draw_best(past_traj, fut_traj, tra_pred, traj_scale, count):
#     print(f"count:{count}")
#     print(f"past_traj:{past_traj.shape}\nfut_traj:{fut_traj.shape}\nx_t:{tra_pred.shape}")
#     past_traj = past_traj[:, :, 2:4]
#     num = fut_traj.size(0)  # num 是场景中的行人数
#     fut_traj = fut_traj[:, 0, :, :].squeeze(1) # (B,12,2)
#
#     # 设置不同的颜色用于区分不同类型的轨迹
#     past_color = 'g'  # 绿色表示past_traj
#     fut_color = 'b'  # 蓝色表示fut_traj
#     x_t_color = 'r'  # 红色表示x_t
#
#     # 创建新的图像和坐标轴对象
#     fig, ax = plt.subplots()
#
#     for j in range(num):  # 遍历每个行人
#         past_traj_1st = past_traj[j].cpu().numpy() * traj_scale  # 8,2
#         fut_traj_1st = fut_traj[j].cpu().numpy() * traj_scale  # 12,2
#         tra_pred_1st = tra_pred[j].cpu().numpy() * traj_scale  # 20,12,2
#
#         # 真实未来轨迹
#         fut_traj_gt = fut_traj_1st
#
#         # 计算每个行人的预测轨迹与真实轨迹之间的平均位移误差
#         distances = []
#         for k in range(tra_pred_1st.shape[0]):  # 对每条预测轨迹（20条）进行计算
#             # 预测轨迹
#             pred_traj = tra_pred_1st[k]
#             # 计算每个时间步的位移误差
#             errors = np.sqrt(np.sum((pred_traj - fut_traj_gt)**2, axis=1))
#             # 计算平均位移误差
#             ade = np.mean(errors)
#             distances.append(ade)
#
#         best_idx = np.argmin(distances)  # 找到平均位移误差最小的轨迹索引
#
#         # 绘制过去的轨迹（每个行人只绘制一次）
#         ax.plot(past_traj_1st[:, 0], past_traj_1st[:, 1], color=past_color)
#         # 绘制真实未来轨迹（每个行人只绘制一次）
#         ax.plot(fut_traj_1st[:, 0], fut_traj_1st[:, 1], color=fut_color)
#         # 只绘制最佳预测轨迹（平均位移误差最小的那条轨迹）
#         ax.plot(tra_pred_1st[best_idx, :, 0], tra_pred_1st[best_idx, :, 1], color=x_t_color)
#
#     # 添加标题、标签等
#     ax.set_title(f'Trajectory Visualization - Index {count} num {num}')
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#
#     # 添加图例
#     ax.legend(['Past Trajectory', 'Future Trajectory', 'Predicted Trajectory'])
#
#     # 保存当前图像
#     plt.savefig(f'results_eth_ucy/cor_fm/5-18-zara2/img-scene/trajectory_{count}.png')
#     plt.close()

def draw_best(past_traj, fut_traj, tra_pred, traj_scale, count):
    print(f"count:{count}")
    print(f"past_traj:{past_traj.shape}\nfut_traj:{fut_traj.shape}\nx_t:{tra_pred.shape}")
    # past_traj = past_traj[:, :, 2:4] # 绝对位置，相对位置，速度分量
    num = fut_traj.size(0)  # num 是场景中的行人数
    fut_traj = fut_traj[:, 0, :, :].squeeze(1) # (B,12,2)

    # 设置不同的颜色用于区分不同类型的轨迹
    past_color = 'g'  # 绿色表示past_traj
    fut_color = 'b'  # 蓝色表示fut_traj
    x_t_color = 'r'  # 红色表示x_t

    # 创建新的图像和坐标轴对象
    fig, ax = plt.subplots()

    for j in range(num):  # 遍历每个行人
        past_traj_1st = past_traj[j].cpu().numpy() * traj_scale  # 8,2
        fut_traj_1st = fut_traj[j].cpu().numpy() * traj_scale  # 12,2
        tra_pred_1st = tra_pred[j].cpu().numpy() * traj_scale  # 20,12,2

        # 真实未来轨迹
        fut_traj_gt = fut_traj_1st

        # 计算每个行人的预测轨迹与真实轨迹之间的平均位移误差
        distances = []
        for k in range(tra_pred_1st.shape[0]):  # 对每条预测轨迹（20条）进行计算
            # 预测轨迹
            pred_traj = tra_pred_1st[k]
            # 计算每个时间步的位移误差
            errors = np.sqrt(np.sum((pred_traj - fut_traj_gt)**2, axis=1))
            # 计算平均位移误差
            ade = np.mean(errors)
            distances.append(ade)

        best_idx = np.argmin(distances)  # 找到平均位移误差最小的轨迹索引

        # 绘制过去的轨迹（每个行人只绘制一次）
        # 使用小圆点表示每个时间点，并连接成线
        ax.plot(past_traj_1st[:, 0], past_traj_1st[:, 1], 'go-', markersize=3, alpha=0.7)  # 'go-' 表示绿色圆点和线

        # 绘制真实未来轨迹（每个行人只绘制一次）
        # 使用小圆点表示每个时间点，并连接成线
        ax.plot(fut_traj_1st[:, 0], fut_traj_1st[:, 1], 'bo-', markersize=3, alpha=0.7)  # 'bo-' 表示蓝色圆点和线

        # 用蓝色小圆点标注过去轨迹的最后一帧
        ax.plot(past_traj_1st[-1, 0], past_traj_1st[-1, 1], 'bo', markersize=3, alpha=0.7)

        # 用蓝色线条连接过去轨迹的最后一帧和未来轨迹的第一帧
        ax.plot([past_traj_1st[-1, 0], fut_traj_1st[0, 0]], [past_traj_1st[-1, 1], fut_traj_1st[0, 1]], 'b-', alpha=0.7)

        # 绘制最佳预测轨迹（平均位移误差最小的那条轨迹）
        # 使用带小点的虚线表示预测轨迹
        ax.plot(tra_pred_1st[best_idx, :, 0], tra_pred_1st[best_idx, :, 1], 'r:.', markersize=3, alpha=0.65)  # 'r:.' 表示红色带点虚线

        # 用红色五角星表示预测轨迹的最后一帧
        ax.plot(tra_pred_1st[best_idx, -1, 0], tra_pred_1st[best_idx, -1, 1], 'r*', markersize=9, alpha=0.65)

        # 用带小点的虚线连接过去轨迹的最后一帧和预测轨迹的第一帧
        ax.plot([past_traj_1st[-1, 0], tra_pred_1st[best_idx, 0, 0]], [past_traj_1st[-1, 1], tra_pred_1st[best_idx, 0, 1]], 'r:.', markersize=3, alpha=0.65)

    # 添加标题、标签等
    ax.set_title(f'Trajectory Visualization - Index {count} num {num}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # 保存当前图像
    plt.savefig(f'results_eth_ucy/cor_fm/5-26-hotel/img-scene/trajectory_{count}.png')
    plt.close()

# def draw_best(past_traj, fut_traj, tra_pred, traj_scale, count):
#     print(f"count:{count}")
#     print(f"past_traj:{past_traj.shape}\nfut_traj:{fut_traj.shape}\nx_t:{tra_pred.shape}")
#
#     num = fut_traj.size(0)  # num 是场景中的行人数
#     fut_traj = fut_traj[:, 0, :, :].squeeze(1) # (B,12,2)
#
#     # 设置不同的颜色用于区分不同类型的轨迹
#     past_color = 'g'  # 绿色表示past_traj
#     fut_color = 'b'  # 蓝色表示fut_traj
#     x_t_color = 'r'  # 红色表示x_t
#
#     # 创建新的图像和坐标轴对象
#     fig, ax = plt.subplots()
#
#     for j in range(num):  # 遍历每个行人
#         past_traj_1st = past_traj[j].cpu().numpy() * traj_scale  # 8,2
#         fut_traj_1st = fut_traj[j].cpu().numpy() * traj_scale  # 12,2
#         tra_pred_1st = tra_pred[j].cpu().numpy() * traj_scale  # 20,12,2
#
#         # 获取真实轨迹的最后一个时间步坐标
#         fut_end = fut_traj_1st[-1]  # 真实未来轨迹的最后一个时间步坐标
#
#         # 计算每个行人的预测轨迹与真实轨迹最后时间步坐标之间的误差
#         distances = []
#         for k in range(tra_pred_1st.shape[0]):  # 对每条预测轨迹（20条）进行计算
#             # 预测轨迹的最后一个时间步坐标
#             pred_end = tra_pred_1st[k, -1]
#             # 计算最终坐标的欧几里得距离
#             error = np.sqrt(np.sum((pred_end - fut_end)**2))  # 计算最终坐标的误差
#             distances.append(error)
#
#         best_idx = np.argmin(distances)  # 找到最小最终坐标误差的轨迹索引
#
#         # 绘制过去的轨迹（每个行人只绘制一次）
#         ax.plot(past_traj_1st[:, 0], past_traj_1st[:, 1], color=past_color)
#         # 绘制真实未来轨迹（每个行人只绘制一次）
#         ax.plot(fut_traj_1st[:, 0], fut_traj_1st[:, 1], color=fut_color)
#         # 只绘制最佳预测轨迹（最终坐标误差最小的那条轨迹）
#         ax.plot(tra_pred_1st[best_idx, :, 0], tra_pred_1st[best_idx, :, 1], color=x_t_color)
#
#     # 添加标题、标签等
#     ax.set_title(f'Trajectory Visualization - Index {count} num {num}')
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#
#     # 添加图例
#     ax.legend(['Past Trajectory', 'Future Trajectory', 'Predicted Trajectory'])
#
#     # 保存当前图像
#     plt.savefig(f'./results/led_augment/try3-13-sdd-leepfrog/images-scene/trajectory_{count}.png')
#     plt.close()