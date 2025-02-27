import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

# 类别和阈值
categories = ["box", "bottle", "can", "cup", "remote", "teapot", "cutlery", "glass", "shoe", "tube"]
thresholds = [0.1, 0.25]

# 第一个算法的数据
values_algo1 = [
    [96.98, 26.36],
    [99.94, 72.10],
    [100.0, 99.49],
    [97.53, 32.96],
    [98.92, 54.16],
    [90.95, 15.27],
    [61.40, 20.25],
    [87.86, 63.51],
    [68.38, 23.11],
    [99.95, 85.73]
]

# 第二个算法的数据
values_algo2 = [
    [91.42, 21.88],
    [97.56, 68.55],
    [100.0, 98.39],
    [96.36, 18.77],
    [90.56, 25.35],
    [96.57, 14.17],
    [64.02, 17.79],
    [96.80, 78.07],
    [55.18, 8.79],
    [100.0, 64.78]
]

# 转换为 DataFrame
df_algo1 = pd.DataFrame(values_algo1, index=categories, columns=thresholds)
df_algo2 = pd.DataFrame(values_algo2, index=categories, columns=thresholds)

# 新的颜色方案（优化的深蓝、深红、浅蓝、浅红）
colors = ["#1F3B75", "#8B1A1A", "#4A90E2", "#E57373"]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))
width = 0.2  # 柱子宽度
x = np.arange(len(categories))  # X 轴位置

# 绘制第一个算法的柱子（深蓝、深红）
for i, threshold in enumerate(thresholds):
    ax.bar(x + i * width, df_algo1[threshold], width, label=f'MK-Pose - IoU$_{{{int(threshold*100)}}}$', alpha=0.8, color=colors[i])

# 绘制第二个算法的柱子（浅蓝、浅红）
for i, threshold in enumerate(thresholds):
    ax.bar(x + i * width + width * 2, df_algo2[threshold], width, label=f'AG-Pose - IoU$_{{{int(threshold*100)}}}$', alpha=0.8, color=colors[i + 2])

# 添加标签和标题
ax.set_xlabel("Category", fontsize=22)
ax.set_ylabel("Value", fontsize=22)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(categories, ha="center", fontsize=22)
ax.tick_params(axis="y", labelsize=22)
ax.legend(fontsize=20)

# 显示图表
plt.savefig(r'E:\aaaaa\research\pose_estimation\mypaper\HouseCat.pdf')