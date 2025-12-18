import pickle
import numpy as np

# 替换为你的PKL文件路径（比如subset是eth，类型是train）
pkl_path = "eth_train.pkl"  # 对应代码保存的 {subset}_{type}.pkl

# 读取PKL文件
with open(pkl_path, 'rb') as f:
    traj_data = pickle.load(f)

# 提取4个核心字段
traj = traj_data['traj']
num_peds_in_seq = traj_data['num_peds_in_seq']
seq_start_end = traj_data['seq_start_end']
frame_list = traj_data['frame_list']

# 打印关键信息
print(f"===== PKL文件核心信息（{pkl_path}）=====")
print(f"1. 总有效行人数量（traj第1维）：{traj.shape[0]}")
print(f"2. 序列长度（观测8帧+预测12帧）：{traj.shape[1]}")
print(f"3. 坐标维度（x,y）：{traj.shape[2]}")
print(f"4. 总序列数：{len(num_peds_in_seq)}")
print(f"5. 每个序列的行人数量：{num_peds_in_seq[:5]}...（前5个序列）")
print(f"6. 每个序列的起止索引：{seq_start_end[:5]}...（前5个序列）")
print(f"7. 每个序列的起始帧号：{frame_list[:5]}...（前5个序列）")

# 可选：查看第一个序列的所有行人轨迹（观测帧+预测帧）
print(f"\n===== 第一个序列的详细信息 =====")
first_seq_start, first_seq_end = seq_start_end[0]
first_seq_traj = traj[first_seq_start:first_seq_end, :, :]  # 第一个序列的所有行人轨迹
print(f"第一个序列的行人数量：{first_seq_end - first_seq_start}")
print(f"第一个序列的轨迹形状（行人数, 20帧, x/y）：{first_seq_traj.shape}")
print(f"第一个行人的8帧观测轨迹（前8帧x,y）：")
print(first_seq_traj[0, :8, :])  # 第0个行人的观测帧（模型输入）
print(f"第一个行人的12帧真实轨迹（后12帧x,y）：")
print(first_seq_traj[0, 8:, :])  # 第0个行人的预测帧（模型标签）