import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

# 类别和阈值
categories = ["box", "bottle", "can", "cup", "remote", "teapot", "cutlery", "glass", "shoe", "tube"]
thresholds = [0.25, 0.5]

# 第一个算法的数据
values_algo1 = [
    [96.73, 40.62],
    [93.54, 66.05],
    [97.37, 56.84],
    [99.56, 66.47],
    [67.68, 15.33],
    [100, 77.85],
    [12.97, 0.2],
    [59.78, 12.21],
    [82.29, 18.93],
    [90.84, 20.33]
]

# 第二个算法的数据
values_algo2 = [
    [61, 4.08],
    [84.79, 27.99],
    [84.28, 14.55],
    [98.55, 52.49],
    [8.63, 0.04],
    [99.74, 75.16],
    [2.49, 0.07],
    [58.91, 1.6],
    [17.63, 0.03],
    [53.98, 3.55]
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