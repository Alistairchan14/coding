import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import numpy as np
import seaborn as sns  # 导入seaborn库
import re



import re



def find_nearest(x, episode):
    # 找到最接近的episode值的索引
    idx = (np.abs(episode - x)).argmin()
    return episode.iloc[idx]

# 设置seaborn风格
sns.set(style='whitegrid')


# 设置字体
plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun']})


# 读取第一个Excel文件
file_path1 = r'D:\\JNU\AApaper\\code\\new\\ress\\MADDPG\\run-.-tag-NO1_C & R_Reward.csv'
data1 = pd.read_csv(file_path1, header=1)  # 跳过表头（第一行）

# 读取第二个Excel文件
file_path2 = r'D:\\JNU\AApaper\\code\\new\\ress\\DDPG\\run-.-tag-NO1_C & R_Reward.csv'
data2 = pd.read_csv(file_path2, header=1)  # 跳过表头（第一行）

# 读取第三个Excel文件
file_path3 = r'D:\\JNU\AApaper\\code\\new\\ress\\DQN\\run-.-tag-NO1_C & R_Reward.csv'
data3 = pd.read_csv(file_path3, header=1)  # 跳过表头（第一行）



# 提取第一个文件的数据
episode1 = data1.iloc[:, 1]  # 第二列是episode
reward1 = data1.iloc[:, 2]   # 第三列是reward

# 提取第二个文件的数据
episode2 = data2.iloc[:, 1]  # 第二列是episode
reward2 = data2.iloc[:, 2]   # 第三列是reward

# 提取第三个文件的数据
episode3 = data3.iloc[:, 1]  # 第二列是episode
reward3 = data3.iloc[:, 2]   # 第三列是reward



# 使用滑动窗口（例如窗口大小=10）来平滑reward数据
window_size = 10
reward_smooth1 = reward1.rolling(window=window_size, center=True).mean()
reward_std1 = reward1.rolling(window=window_size, center=True).std()

reward_smooth2 = reward2.rolling(window=window_size, center=True).mean()
reward_std2 = reward2.rolling(window=window_size, center=True).std()

reward_smooth3 = reward3.rolling(window=window_size, center=True).mean()
reward_std3 = reward3.rolling(window=window_size, center=True).std()


# 标出要标记的蓝色点（在第一条曲线上，x值为给定的列表）
red_x = red_x = [119,189,199,226,238,248,268,301,316,326,336,346,367,387,397,407,420,596,629,639,688,705,724,734,752,765,775,792,819,831,852,862,872,882,892,902,914,924,934,944,954,971,981,991,1011,1021,1036,1046,1056,1066,1088,1103,1114,1124,1146,1156,1167,1187,1198,1216,1228,1238,1249,1259,1278,1288,1301,1302,1303,1304,1305,1306,1316,1345,1378,1388,1400,1412,1422,1467,1487,1542,1559,1773,2131,2430,2440,2453,2465,2477,2490,2500,2511,2524,2534,2544,2554,2564,2581,2591,2610,2641,2656,2669,2679,2690,2705,2776,2777,2778,2779,2780,2781,2782,2930,2931,2932,2933,2934,2935,2936,2937,2938,2939,2940,2941,2942,3032,3033,3034]

#red_x = [113,9712]

# 标出要标记的红色点（在第二条曲线上，x值为给定的列表）
blue_x = []

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制第一个文件的数据
ax.plot(episode1, reward_smooth1, color='#d62447', lw=2, zorder=3, label='MADDPG algorithm')

ax.fill_between(episode1, reward_smooth1 - reward_std1, reward_smooth1 + reward_std1, 
                color='#d62447', alpha=0.15, zorder=1)

# 绘制第二个文件的数据
ax.plot(episode2, reward_smooth2, color='#091aa8', lw=1.5, zorder=2, label='DDPG algorithm', linestyle='-')
ax.fill_between(episode2, reward_smooth2 - reward_std2, reward_smooth2 + reward_std2, 
                color='#091aa8', alpha=0.15, zorder=1)

# 绘制第三个文件的数据
ax.plot(episode3, reward_smooth3, color='#5facad', lw=1.5, zorder=1, label='DQN algorithm', linestyle='-')
ax.fill_between(episode3, reward_smooth3 - reward_std3, reward_smooth3 + reward_std3, 
                color='#5facad', alpha=0.15, zorder=1)


# 标出红色的点（对应给定的x值）
red_rewards_valid = []
red_x_valid = []

for x in red_x:
    if x in episode1.values:  # 如果 x 直接存在
        red_x_valid.append(x)
        red_rewards_valid.append(reward_smooth1[episode1 == x].values[0])
    else:
        # 如果找不到 x 值，就近搜索
        nearest_x = find_nearest(x, episode1)
        red_x_valid.append(nearest_x)
        red_rewards_valid.append(reward_smooth1[episode1 == nearest_x].values[0])

# 标出红色的点
ax.scatter(red_x_valid, red_rewards_valid, facecolors='white', edgecolors='red', zorder=3, label='the convergence points of MADDPG algorithm', s=100, marker='o', linewidths=2)

# 标出蓝色的点（对应给定的x值）
blue_rewards_valid = []
blue_x_valid = []

for x in blue_x:
    if x in episode2.values:  # 如果 x 直接存在
        blue_x_valid.append(x)
        blue_rewards_valid.append(reward_smooth2[episode2 == x].values[0])
    else:
        # 如果找不到 x 值，就近搜索
        nearest_x = find_nearest(x, episode2)
        blue_x_valid.append(nearest_x)
        blue_rewards_valid.append(reward_smooth2[episode2 == nearest_x].values[0])

# 标出蓝色的点
#ax.scatter(blue_x_valid, blue_rewards_valid, color='blue', zorder=3, label='DDPG算法（不考虑维修）收敛点', s=50, marker='^')



# 设置轴标签和标题
ax.set_xlabel('Episode', fontsize=20)
ax.set_ylabel('Reward', fontsize=20)
#ax.set_title('Reinforcement Learning Training Map', fontsize=16)

# 设置图例并将字体设置为Times New Roman
ax.legend(loc='lower right', fontsize=14)

# 设置字体
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 调整边距
plt.tight_layout()

plt.savefig('C:/Users/yan/Desktop/paper pic/huiyi/case_train_reward.png', dpi=600)
# 显示图形
plt.show()
