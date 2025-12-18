import numpy as np

# 替换为你的文件路径
train_path = "nba_train.npy"
data = np.load(train_path)

# 提取第一个样本的结构
sample_0 = data[0]
print("第一个样本的形状（时间步, 实体数, 坐标维度）：", sample_0.shape)

# 提取第一个样本第一帧的所有实体坐标
frame_0 = sample_0[0]
print("第一帧11个实体的坐标预览：\n", frame_0[:3])  # 打印前3个实体的(x,y)