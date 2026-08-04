
import numpy as np
import matplotlib.pyplot as plt

# 读取数据文件
data = []
with open('build/bin/qerror_ppc_non_adaptive.txt', 'r') as f:
    for line in f:
        # 跳过空行和注释行
        if line.strip() and not line.startswith('#'):
            data.append(list(map(float, line.split())))

# 转换为NumPy数组
data = np.array(data)
# 提取时间和六个自由度的误差
time = data[:, 0]  # 第一列是时间
joint_errors = data[:, 1:7]  # 后六列是六个关节的误差

#joint_errors = joint_errors[len(time)-500:len(time),:]
#time = time[len(time)-500:len(time)]

# 关节名称（根据自由度命名）
joint_names = [
    "Joint 1 ",
    "Joint 2 ",
    "Joint 3 ",
    "Joint 4 ",
    "Joint 5 ",
    "Joint 6 ",
]

# 创建2x3的子图布局
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Robotic Arm Joint Errors', fontsize=20)

# 绘制每个关节的误差曲线
for i, ax in enumerate(axes.flat):
    if i < 6:  # 确保只绘制6个关节
        # 绘制误差曲线
        ax.plot(time, joint_errors[:, i], 'b-', linewidth=1.5)
        
        # 设置标题和标签
        ax.set_title(joint_names[i], fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Error (rad/m)', fontsize=12)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 自动调整Y轴范围，但保留10%的边距
        y_min = np.min(joint_errors[:, i])
        y_max = np.max(joint_errors[:, i])
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
plt.show()

# 可选：保存图像到文件
# plt.savefig('joint_errors.png', dpi=300, bbox_inches='tight')