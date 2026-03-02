import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# 环境导入
from environment6 import Environment 

env = Environment(num_customers = 6, num_UAVs = 2)  # 假设这里是你的自定义环境

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUDA_VISIBLE_DEVICES = 0

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) #队列，先进先出
    
    def add(self, UAV_obs, maintenance_action, routing_action, reward, next_UAV_obs, done, selected_UAV): 
        experience = (UAV_obs, maintenance_action, routing_action, reward, next_UAV_obs, done, selected_UAV)
        self.buffer.append(experience)

    def sample(self, batch_size):
        # 生成一个随机索引列表
        indices = torch.randint(0, len(self.buffer), (batch_size,), device=device)

        # 使用索引来选择数据
        transitions = [self.buffer[i] for i in indices]

        # 拆解 transitions
        unpacked = list(zip(*transitions))

        # 将数据转换为torch.Tensor并移动到GPU（如果有的话）
        UAV_obs = torch.stack(unpacked[0]).float().detach().to(device)
        maintenance_action = torch.stack(unpacked[1]).float().detach().to(device)
        routing_action = torch.stack(unpacked[2]).long().detach().to(device)
        reward = torch.tensor(unpacked[3], dtype=torch.float32).detach().to(device)
        next_UAV_obs = torch.stack(unpacked[4]).float().detach().to(device)
        done = torch.tensor(unpacked[5], dtype=torch.float32).detach().to(device)
        selected_UAV = torch.tensor(unpacked[6]).long().detach().to(device)


        return (
            UAV_obs, 
            maintenance_action, 
            routing_action, 
            reward, 
            next_UAV_obs, 
            done, 
            selected_UAV
        )
    
    def size(self):
        return len(self.buffer)
    
class Qnetwork(nn.Module):
    def __init__(self, obs_dim, num_customers):
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.routing_out = nn.Linear(128, num_customers + 1)  # 仓库+客户点
        self.maintenance_mean = nn.Linear(128, 1)
        self.maintenance_std = nn.Linear(128, 1)

    def forward(self, UAV_obs_tensor_i):
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)
        x = UAV_obs_tensor_i
        x = F.relu(self.fc1(x))
        mean = self.maintenance_mean(x)
        std = torch.exp(self.maintenance_std(x))

        masks_tensor = self.get_UAV_action_mask(UAV_obs_tensor_i)
        masks_tensor = masks_tensor.to(device)

        routing_q_values = self.routing_out(x)
        routing_q_values = routing_q_values * masks_tensor + ~masks_tensor * (-1e9)

        
        return mean, std, routing_q_values
    
    def get_UAV_action_mask(self, UAV_obs_tensor_i):
        
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)
        # 判断 UAV_obs_tensor_i 是否是二维数组（用于区分单样本和多样本情况）
        is_batch = len(UAV_obs_tensor_i.shape) == 2  # 如果是二维，则是批量样本
        if is_batch:
            num_batch, num_feature = UAV_obs_tensor_i.shape  # 获取样本数量和每个样本的维度
        else:
            num_batch = 1  # 如果是一维，则是单样本
            UAV_obs_tensor_i = UAV_obs_tensor_i.unsqueeze(0).to(device)  # 将单样本转换为二维，方便后续操作
            num_feature = UAV_obs_tensor_i.shape[1]  # 获取样本的维度

        # 从样本中提取任务序列（最后 6 个元素）
        task_sequences = UAV_obs_tensor_i[:, -num_customers:]

        # 初始化 masks，第一列为 True（表示可以选择返回仓库），其余列为 False
        masks = torch.zeros((num_batch, num_customers + 1), dtype=torch.bool).to(device)  # 创建布尔张量
        masks[:, 0] = True  # 设置第一个动作（返回仓库）为 True

        # 从 UAV_obs_tensor_i 中提取关键信息
        energy = UAV_obs_tensor_i[:, 4].to(device) # 提取电量信息
        broken = UAV_obs_tensor_i[:, 6].to(device) # 提取是否损坏信息
        load = UAV_obs_tensor_i[:, 3].to(device) # 提取当前载重信息
        destination = UAV_obs_tensor_i[:, 1].long().to(device) # 提取当前目的地（整数形式）

        # 从环境变量 `env` 中获取需求量和距离矩阵
        customer_cargo_demands = torch.tensor(env.customer_cargo_demands).to(device)  # 获取每个客户点的货物需求量
        distance_matrix = torch.tensor(env.distance_matrix).to(device)  # 获取距离矩阵
        max_UAVs_load = torch.tensor(env.max_UAVs_load).to(device)  # 获取无人机的最大载重
        fix_v = torch.tensor(env.fix_v).to(device)  # 获取固定飞行速度
        energy_parameter = torch.tensor(env.config['energy_parameter']).to(device)  # 获取能量参数
        max_energy = torch.tensor(env.max_energy).to(device)  # 获取最大能量

        # 屏蔽损坏的无人机：
        # 若某无人机损坏（broken=1），则只允许返回仓库（第1列为True，其余列为False）
        masks[broken == 1] = torch.tensor([True] + [False] * num_customers, dtype=torch.bool).to(device)

        # 根据任务序列更新与客户点对应的 masks
        masks[:, 1:] = (task_sequences == 1)  # 设置任务序列为 1 的客户点对应的动作为 True

        # 如果上一轮目的地是仓库，则屏蔽仓库动作
        masks[destination == 0, 0] = False

        # 遍历客户点，并基于载重和电量约束更新 masks
        for j in range(1, num_customers + 1):  # 客户点索引从 1 到 6
            demand_j = customer_cargo_demands[j - 1]  # 获取第 j 个客户点的需求量

            # 计算从当前目的地到客户点 j 的到达时间
            arrival_time_to_j = distance_matrix[destination, j] / fix_v
            # 计算从客户点 j 返回仓库的时间
            arrival_time_to_warehouse = distance_matrix[j, 0] / fix_v

            # 计算到客户点 j 和返回仓库的总载重
            load_to_j = (100 + load + demand_j) ** (3 / 2)
            # 计算到客户点 j 的剩余电量
            energy_to_j = energy - energy_parameter * load_to_j * arrival_time_to_j * 0.001 // max_energy
            # 计算从客户点 j 返回仓库后的剩余电量
            energy_to_warehouse = energy_to_j - energy_parameter * load_to_j * arrival_time_to_warehouse * 0.001 // max_energy

            # 如果载重超限或电量不足，屏蔽对应客户点的动作
            masks[:, j] &= (load + demand_j <= max_UAVs_load) & (energy_to_warehouse > 0)

        # 将 masks 调整为适配单样本和多样本的格式
        if is_batch:
            # 如果是多样本，masks 保持二维
            masks = masks  # [num_batch, 7]
        else:
            # 如果是单样本，masks 转换为 1 维
            masks = masks.squeeze(0)  # [7]

        masks_tensor = masks.to(device)

        return masks_tensor
    
class DQN:
    def __init__(self, obs_dim, num_customers, lr=5e-5, gamma=0.95, epsilon=0.1, target_update=10, buffer_capacity=5000, batch_size=256):
        self.obs_dim = obs_dim
        self.num_customers = num_customers
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.q_network = Qnetwork(obs_dim, num_customers).to(device)
        self.target_q_network = Qnetwork(obs_dim, num_customers).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def take_action(self, UAV_obs_tensor_i, selected_UAV):
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)

        mean, std, routing_q_values = self.q_network(UAV_obs_tensor_i)
        maintenance_action = torch.normal(mean, std).clamp(0, 1).to(device)  # 采样并限制在 [0, 1] 范围内
        
        if torch.rand(1).item() < self.epsilon:  # 使用 PyTorch 的随机数生成

            valid_q_values = routing_q_values[routing_q_values != -1e9] # 过滤掉mask后routing Q值中 -1e9 的无效值
            random_q_value = valid_q_values[torch.randint(len(valid_q_values), (1,))].item() # 随机选择一个有效Q值
            routing_action = (routing_q_values == random_q_value).nonzero(as_tuple=True)[0].item() # 找到该随机值对应的索引即为路径决策
            routing_action = torch.tensor(routing_action).unsqueeze(0).to(device)

        else:
            with torch.no_grad():
                # 选取最大 q_value 对应的动作
                routing_action = torch.argmax(routing_q_values, dim=-1).unsqueeze(0).to(device)  # 返回一个 1 维张量
        
        # 获取状态
        UAV = torch.tensor(env.UAV_obs_matrix, dtype=torch.float32).to(device)
        selected_UAV = int(selected_UAV.item())
        broken = UAV[selected_UAV][6].to(device)
        routing_action_value = routing_action.item()

        # 维修决策的限制条件
        if broken == 1:  # 如果无人机损坏，维修动作必须为0（完全换新）
            maintenance_action = torch.tensor([0.0], dtype=torch.float32, device=device)
        elif routing_action_value != 0:  # 如果routing_action不是返回仓库，维修动作必须为1（不维修）
            maintenance_action = torch.tensor([1.0], dtype=torch.float32, device=device)
        elif routing_action_value == 0:  # 如果routing_action是返回仓库，根据actor输出值选择维修回退因子
            maintenance_action = maintenance_action

        return maintenance_action, routing_action
    
    def update(self, UAV_obs, maintenance_action, routing_action, rewards, next_UAV_obs, done, selected_UAV):

        transition_dict = {
            'UAV_obs': UAV_obs,
            'maintenance_action': maintenance_action,
            'routing_action': routing_action,
            'rewards': rewards,
            'next_UAV_obs': next_UAV_obs,
            'done': done,
            'selected_UAV': selected_UAV
        }
        
        UAV_obs = transition_dict['UAV_obs']
        if isinstance(UAV_obs, torch.Tensor):
            UAV_obs = UAV_obs.clone().detach().to(device)
        else:
            UAV_obs = torch.tensor(UAV_obs, dtype=torch.float32).to(device)
            
        maintenance_action = transition_dict['maintenance_action'].clone().detach().to(device).float()
        routing_action = transition_dict['routing_action'].clone().detach().to(device).float()

        rewards = transition_dict['rewards']
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.clone().detach().to(device)
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        next_UAV_obs = transition_dict['next_UAV_obs']
        if isinstance(next_UAV_obs, torch.Tensor):
            next_UAV_obs = next_UAV_obs.clone().detach().to(device)
        else:
            next_UAV_obs = torch.tensor(next_UAV_obs, dtype=torch.float32).to(device)

        done = transition_dict['done']
        if isinstance(done, torch.Tensor):
            done = done.clone().detach().to(device)
        else:
            done = torch.tensor(done, dtype=torch.float32).to(device)

        selected_UAV = transition_dict['selected_UAV']
        if isinstance(selected_UAV, torch.Tensor):
            selected_UAV = selected_UAV.clone().detach().to(device)
        else:
            selected_UAV = torch.tensor(selected_UAV, dtype=torch.long).unsqueeze(0).to(device)

        next_UAV_obs_i = next_UAV_obs[torch.arange(next_UAV_obs.size(0)), selected_UAV.long()].clone().to(device)
        # 下个状态的最大Q值
        _, _, max_next_q_values = self.target_q_network(next_UAV_obs_i)
        max_next_q_values = max_next_q_values.max(1)[0].view(-1, 1).to(device)

        rewards = rewards.view(-1, 1).to(device)  # 将 rewards 从 [256] 调整为 [256, 1]
        done = done.view(-1, 1).to(device)        # 将 done 从 [256] 调整为 [256, 1]
        # 计算 TD 目标 (target Q values)，结合 maintenance 和 routing 动作的目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - done).to(device)  # TD误差目标

        # 计算 DQN 损失（均方误差损失）

        # routing_action 的损失计算
        _, _, routing_q_values = self.q_network(UAV_obs[torch.arange(self.batch_size), selected_UAV.long().squeeze(), :])
        routing_action = routing_action.to(torch.long)
        routing_q_values = routing_q_values.gather(1, routing_action).to(device)
        dqn_loss = nn.MSELoss()(routing_q_values, q_targets)

        # 清除梯度并进行反向传播
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()  # 更新网络参数

        if self.count % self.target_update == 0:
            self.target_q_network.load_state_dict(
                self.q_network.state_dict())  # 更新目标网络
        self.count += 1

        return dqn_loss.item()




#初始化经验回放池
replay_buffer = ReplayBuffer(capacity = 5000)

#初始化环境和算法
obs_dim = env.observation_space.shape[0]  # 观测维度
num_customers = env.num_customers  # 客户数量
num_UAVs = env.num_UAVs  # 无人机数量
maintenance_action_dim = 1  
routing_action_dim = 1  
selected_UAV_dim = 1
samp_number_dim = 1

dqn = DQN(obs_dim, num_customers)
recent_costs = []
samp_number = 1  # 初始采样编号
max_samp_number = 1200  # 最大采样编号

# 初始化每台无人机的路径和总reward存储
UAV_routes = [[] for _ in range(num_UAVs)]  # 存储每台无人机的路径
UAV_rewards = [0 for _ in range(num_UAVs)]  # 存储每台无人机的reward
total_reward = 0
episode_rewards = [] # 存储每个episode的reward

# 创建SummaryWriter对象来记录数据
log_dir = 'D:\\JNU\\AApaper\\code\\new\\DQN_result\\tensorboard'
writer = SummaryWriter(log_dir=log_dir)
#输入这个：tensorboard --logdir=D:\JNU\AApaper\code\somepic

# 指定日志文件路径
log_file_path = 'D:\\JNU\\AApaper\\code\\new\\DQN_result\\obs.txt'
# 新的网络参数日志路径（每50代保存一次）
network_log_file_path = 'D:\\JNU\\AApaper\\code\\new\\DQN_result\\para.txt'
seed_log_file_path = 'D:\\JNU\\AApaper\\code\\new\\DQN_result\\seed.txt'

# 如果文件已存在，清空文件内容
if os.path.exists(log_file_path):
    open(log_file_path, 'w').close()
if os.path.exists(network_log_file_path):
    open(network_log_file_path, 'w').close()
if os.path.exists(seed_log_file_path):
    open(seed_log_file_path, 'w').close()

# 添加更新间隔变量
update_interval = 50
step_counter = 0

#添加记录变量
record_interval = 50  # 每 50 代记录一次参数
clear_interval = 1   # 每记录 1 次清空文件
record_count = 0  # 用于记录已保存的次数

#开始训练
num_episodes = 60
for episode in range(num_episodes):
    UAV_rewards = [0 for _ in range(num_UAVs)]  # 每个episode开始时，重置每台无人机的reward
    UAV_routes = [[] for _ in range(num_UAVs)]  # 存储每台无人机的路径
    UAV_costs = [0 for _ in range(num_UAVs)]  # 每台无人机的成本初始化
    episode_reward = 0  # 每个episode的总奖励
    episode_cost = 0  # 初始化每个 episode 的总成本
    episode_critic_loss = 0  # 每个episode的总critic损失
    episode_actor_loss = 0  # 每个episode的总actor损失
    UAV_obs = env.reset()
    done = False

    # 提取x,y,d分别是客户点的x坐标、y坐标和需求量
    x, y, d = env.customer_list()

    while not done:
        #把xyd转换为tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).squeeze().to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).squeeze().to(device)
        d_tensor = torch.tensor(d, dtype=torch.float32).squeeze().to(device)

        selected_UAV = env.selected_UAV()
        #把selected_UAV转换为tensor
        selected_UAV_tensor = torch.tensor(selected_UAV, dtype=torch.float32).unsqueeze(0).to(device)

        # 获取obs
        UAV_obs_array = np.array(UAV_obs)
        UAV_obs_tensor = torch.tensor(UAV_obs_array, dtype=torch.float32).to(device)
        UAV_obs_tensor_i = UAV_obs_tensor[int(selected_UAV_tensor.item())].clone().to(device)

        # 根据obs选择动作
        maintenance_action, routing_action = dqn.take_action(UAV_obs_tensor_i, selected_UAV_tensor)

        # 执行一步动作
        next_UAV_obs, rewards, done, cost = env.step(maintenance_action, routing_action , selected_UAV)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

        # 把next_UAV_obs转换为tensor
        next_UAV_obs_array = np.array(next_UAV_obs)
        next_UAV_obs_tensor = torch.tensor(next_UAV_obs_array, dtype=torch.float32).to(device)

        # 将experience加入到ReplayBuffer
        replay_buffer.add(UAV_obs_tensor, maintenance_action, routing_action, \
                          rewards_tensor, next_UAV_obs_tensor, done_tensor, selected_UAV_tensor)
        
        # 每隔固定步数（50步）更新dqn
        if step_counter % update_interval == 0 and replay_buffer.size() >= dqn.batch_size:  # 确保buffer中有足够的数据
            batch = replay_buffer.sample(dqn.batch_size)
            UAV_obs_batch, maintenance_action_batch, routing_action_batch, rewards_batch, \
                next_UAV_obs_batch, done_batch, selected_UAV_batch = batch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = tuple(t.to(device) for t in batch)  # 将所有批量数据转移到设备

            transition_dict = {
                'UAV_obs': batch[0],
                'maintenance_action': batch[1],
                'routing_action': batch[2],
                'rewards': batch[3],
                'next_UAV_obs': batch[4],
                'done': batch[5],
                'selected_UAV': batch[6]
            }

            dqn_loss = dqn.update(**transition_dict)

            writer.add_scalar('NO3_DQN Loss', dqn_loss, step_counter)  # 记录critic_loss

        # 获取每台无人机的当前位置和目的地
        UAV = env.UAV_obs_matrix
        if isinstance(selected_UAV, int):
            pass
        elif isinstance(selected_UAV, torch.Tensor):
            selected_UAV = int(selected_UAV.item())
        current_position = int(UAV[selected_UAV][0])  # 假设有这样的字段存储位置
        destination = int(UAV[selected_UAV][1])  # 目的地
        
        # 更新选中无人机的路径
        UAV_routes[selected_UAV].append(destination)  # 记录选中无人机的目的地
        UAV_rewards[selected_UAV] += rewards  # 更新该无人机的reward
        UAV_costs[selected_UAV] += cost  # 更新该无人机的成本
        
        # 打印每台无人机的当前路径和reward
        print(f'UAV{selected_UAV + 1} routes: {"-".join(map(str, UAV_routes[selected_UAV]))}, reward: {UAV_rewards[selected_UAV]}')

        UAV_obs = next_UAV_obs

        # 增加步数计数器
        step_counter += 1

        # 计算并打印总reward
        total_reward = sum(UAV_rewards)
        total_cost = sum(UAV_costs)

    episode_reward = total_reward
    episode_cost = total_cost
    episode_reward_b = -total_cost

    for i, reward in enumerate(UAV_rewards):
        #writer.add_scalar(f'UAV_Rewards/UAV{i+1}', reward, episode)
        writer.add_scalars(
            'NO2_UAV_Rewards',  # 图表名称
            {f'UAV{i+1}': reward for i, reward in enumerate(UAV_rewards)},  # 奖励字典
            episode  # 当前轮次作为横轴
        )

    # 记录reward、critic_loss和actor_loss到TensorBoard
    writer.add_scalar('NO1_C & R/Reward', episode_reward, episode)
    writer.add_scalar('NO1_C & R/Cost', episode_cost, episode)
    writer.add_scalar('NO1_C & R/Reward_b', episode_reward_b, episode)

    print(f"Episode {episode} finished. Total Reward: {episode_reward}, Cost:{episode_cost.item():.4f}")

    # 输出每台无人机的路径和成本到日志文件
    with open(log_file_path, 'a') as f:
        f.write(f"Episode {episode}:\n")
        f.write(f"Customer_x:{x}:\n")
        f.write(f"Customer_y:{y}:\n")
        f.write(f"Customer_d:{d}:\n")
        f.write(f"UAV_obs:{UAV_obs}:\n")
        for i in range(num_UAVs):
            f.write(f"UAV{i + 1} routes: {'-'.join(map(str, UAV_routes[i]))}, cost: {UAV_rewards[i]}\n")
        f.write(f"Total reward: {total_reward}\n")
        f.write(f"Total cost: {total_cost}\n")
    
    # 每50代记录Actor和Critic网络的参数
    if (episode + 1) % record_interval == 0:
        # 更新记录计数
        record_count += 1

        # 判断是否需要清空文件
        if record_count > clear_interval:
            record_count = 1  # 重置记录计数
            write_mode = 'w'  # 清空文件再写入
        else:
            write_mode = 'a'  # 追加记录

        with open(network_log_file_path, write_mode) as f:
            f.write(f"Episode {episode + 1} Network Parameters:\n")
    
            # 记录 Q 网络的参数
            for i, param in enumerate(dqn.q_network.parameters()):  # 这里假设 Q 网络存储在 agent.q_network
                f.write(f"QNetwork Param {i}: {param.data.tolist()}\n")

    # 将 episode_cost 添加到 recent_costs 列表中
    recent_costs.append(episode_cost)
    if len(recent_costs) > 10:  # 保证 recent_costs 的长度为 10
        recent_costs.pop(0)

    # 检查最近 10 个 cost 是否收敛
    if len(recent_costs) == 10 and (max(recent_costs) - min(recent_costs) <= 0.001):
        print(f"第{samp_number}批客户在第 {episode} 次 episode 收敛，开始下一轮训练。")

        # 记录收敛的 episode 到日志文件
        with open(seed_log_file_path, 'a') as f:
            f.write(f"第{samp_number}批客户在第 {episode} 次 episode 收敛，重置客户点信息进行下一轮训练。\n")

        env.update_seed()  # 调用 update_seed 更新客户点
        samp_number += 1   # 更新采样编号

        # 重置 recent_costs，用于下一轮收敛判断
        recent_costs = []

# 训练结束后关闭TensorBoard writer
writer.close()

# 最终打印每台无人机的完整飞行路线和总reward
for i in range(num_UAVs):
    print(f'UAV{i + 1} routes: {"-".join(map(str, UAV_routes[i]))}, UAV{i + 1} reward: {UAV_rewards[i]}')

print(f'Total reward: {total_reward}')
print(f'Episode rewards: {episode_rewards}')

# 假设 self.actor 是你的 Actor 网络实例
save_dir = 'D:/saved_models'  # 保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 如果目录不存在，则创建

save_path = os.path.join(save_dir, 'q_network_full.pth')  # 完整文件路径

# 保存整个模型
torch.save(dqn.q_network, save_path)
print(f"模型已保存到：{save_path}")