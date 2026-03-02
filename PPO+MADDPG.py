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
#pip install scikit-learn
import sklearn
from sklearn.mixture import GaussianMixture
from torch.distributions import Categorical

# 环境导入
from environment_up import Environment 

env = Environment(num_UAVs = 3, num_customers = 10)  # 使用10个客户点

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUDA_VISIBLE_DEVICES = 0


# 定义Actor网络，输出维修action和路径action
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, num_customers):
        super(ActorNetwork, self).__init__()
        
        # 公共的前两层隐藏层
        self.fc1 = nn.Linear(obs_dim + num_customers * 3, 128)
        self.fc2 = nn.Linear(128, 128)

        # 维修action部分（连续输出，范围在0到1之间）
        self.maintenance_out = nn.Linear(128, 1)
        self.maintenance_activation = nn.Sigmoid()  # 确保输出在[0, 1]

        # 路径action部分（输出每个点的概率分布）
        self.routing_out = nn.Linear(128, num_customers + 1)  # 仓库+客户点

    def forward(self, x_tensor, y_tensor, d_tensor, UAV_obs_tensor_i, selected_UAV):
        
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        d_tensor = d_tensor.to(device)
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)
        masks_tensor = self.get_UAV_action_mask(UAV_obs_tensor_i)
        masks_tensor = masks_tensor.to(device)
        
        x = torch.cat((x_tensor, y_tensor, d_tensor, UAV_obs_tensor_i), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # 输出维修action
        maintenance_action = self.maintenance_activation(self.maintenance_out(x))
        
        # 输出路径action，并通过mask和softmax计算最终概率
        routing_logits = self.routing_out(x)

        #routing_logits = routing_logits * masks_tensor + (1 - masks_tensor) * (-1e9)
        routing_logits = routing_logits * masks_tensor + ~masks_tensor * (-1e9)

        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        return maintenance_action, routing_probs
    
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
        customer_sequences = UAV_obs_tensor_i[:, -num_customers:]

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
        masks[:, 1:] = (customer_sequences == 1)  # 设置任务序列为 1 的客户点对应的动作为 True

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
            energy_to_j = energy - energy_parameter * load_to_j * arrival_time_to_j * 0.1 / max_energy
            # 计算从客户点 j 返回仓库后的剩余电量
            energy_to_warehouse = energy_to_j - energy_parameter * load_to_j * arrival_time_to_warehouse * 0.1 / max_energy

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

# 定义Critic网络，用于Q值估计
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, maintenance_action_dim, routing_action_dim, selected_UAV_dim, num_customers):
        super(CriticNetwork, self).__init__()
        
        # 输入为观测obs和动作action的组合
        self.fc1 = nn.Linear(obs_dim * num_UAVs + maintenance_action_dim + routing_action_dim + selected_UAV_dim + num_customers * 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)  # 输出Q值

    def forward(self, x_tensor, y_tensor, d_tensor, UAV_obs, maintenance_action, routing_action, selected_UAV):

        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        d_tensor = d_tensor.to(device)
        UAV_obs = UAV_obs.view(256, -1).to(device)
        maintenance_action = maintenance_action.to(device)
        routing_action = routing_action.to(device)
        selected_UAV = selected_UAV.to(device)

        x = torch.cat([x_tensor, y_tensor, d_tensor, UAV_obs, maintenance_action, routing_action, selected_UAV], dim = -1)  # 将观测和动作拼接
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value

# 定义经验回放池 ReplayBuffer
class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity  # 缓冲区最大容量
        self.buffer = []  # 使用列表来存储经验
        self.samp_counts = {}  # 记录每个编号的经验数量，键为编号，值为数量

    def add(self, samp_number_tensor, x_tensor, y_tensor, d_tensor, UVA_obs, maintenance_action, routing_action, reward, next_UAV_obs, done, selected_UAV):  # 将数据加入 buffer
        experience = (samp_number_tensor, x_tensor, y_tensor, d_tensor, UVA_obs, maintenance_action, routing_action, reward, next_UAV_obs, done, selected_UAV)
        samp_number = experience[0]  # 编号是 experience 的第一项
        if len(self.buffer) < self.capacity:
            # 缓冲区未满时直接添加
            self.buffer.append(experience)
            # 更新编号对应的经验数量
            self.samp_counts[samp_number] = self.samp_counts.get(samp_number, 0) + 1
        else:
            # 缓冲区满时，先检查是否满足每组经验不少于 200 条
            removable = [
                exp for exp in self.buffer
                if self.samp_counts[exp[0]] > 200
            ]
            if removable:
                # 按照 reward 最小的经验删除
                removable.sort(key=lambda exp: exp[7])  # 经验的第 7 项是 reward
                to_remove = removable[0] # 找到 reward 最小的经验
                self.buffer.remove(to_remove) # 从缓冲区中删除该经验
                self.samp_counts[to_remove[0]] -= 1 # 更新编号对应的经验数量
            self.buffer.append(experience) # 添加新的经验
            self.samp_counts[samp_number] = self.samp_counts.get(samp_number, 0) + 1 # 更新编号的经验数量

    def sample(self, batch_size):
        # 获取所有编号（samp_number）以及对应的样本
        samp_numbers = [exp[0].item() for exp in self.buffer]  # 提取所有经验的编号
        unique_numbers = list(set(samp_numbers))  # 获取所有唯一的编号
        samples_per_number = {key: [] for key in unique_numbers}  # 初始化每个编号的样本存储字典
        for exp in self.buffer:
            samples_per_number[exp[0].item()].append(exp)  # 将经验按照编号分类

        # 使用高斯混合模型（GMM）拟合编号分布
        gmm = GaussianMixture(n_components=len(unique_numbers), random_state=42)  # 初始化 GMM 模型
        samp_numbers_array = np.array(samp_numbers).reshape(-1, 1)  # 将编号转为二维数组，用于 GMM 拟合
        gmm.fit(samp_numbers_array)  # 使用 GMM 对编号进行拟合

        # 根据 GMM 计算每组编号的抽样比例
        densities = gmm.predict_proba(samp_numbers_array)  # 计算每个编号的概率密度
        sampling_ratios = densities.mean(axis=0)  # 计算每组编号的平均密度(axis=0表示对列求均值)
        sampling_ratios /= sampling_ratios.sum()  # 对抽样比例进行归一化，确保总和为 1

        # 按比例从每组编号中随机采样
        sampled_experiences = []  # 存储最终的采样结果

        # 确保所有组的样本数量满足总数为 batch_size
        group_sample_counts = [int(batch_size * ratio) for ratio in sampling_ratios]
        group_sample_counts[-1] += batch_size - sum(group_sample_counts)

        for group_idx, (number, samples) in enumerate(samples_per_number.items()):
            group_sample_count = group_sample_counts[group_idx]  # 获取调整后的样本数量

            # 确保所有样本为张量并转移到 GPU/CPU
            samples = [
                [torch.tensor(item, dtype=torch.float32).to(device) if not isinstance(item, torch.Tensor) else item.to(device)
                for item in exp]
                for exp in samples
            ]
            
            # 使用 PyTorch 的随机采样方法
            indices = torch.randperm(len(samples))[:min(group_sample_count, len(samples))]
            sampled_experiences.extend([samples[i] for i in indices])

        # 如果最终数量不足，补充样本
        if len(sampled_experiences) < batch_size:
            samples_fill = batch_size - len(sampled_experiences)
            indices_fill = torch.randperm(len(self.buffer))[:samples_fill]
            sampled_experiences.extend([self.buffer[i] for i in indices_fill])

        # 从采样结果中随机抽取 batch_size 个样本
        sampled_indices = torch.randperm(len(sampled_experiences))[:batch_size]
        sampled_experiences = [sampled_experiences[i] for i in sampled_indices]

        # 将采样的经验解压为各个字段
        unpacked = list(zip(*sampled_experiences))  # 解压采样结果
        samp_number = torch.stack(unpacked[0]).long().detach().to(device)  # 转为张量
        x_tensor = torch.stack(unpacked[1]).float().detach().to(device)  # 堆叠张量
        y_tensor = torch.stack(unpacked[2]).float().detach().to(device)  # 堆叠张量
        d_tensor = torch.stack(unpacked[3]).float().detach().to(device)  # 堆叠张量
        UAV_obs = torch.stack(unpacked[4]).float().detach().to(device)  # 堆叠张量
        maintenance_action = torch.stack(unpacked[5]).float().detach().to(device)  # 转为张量并扩展维度
        routing_action = torch.stack(unpacked[6]).long().detach().to(device)  # 转为张量并扩展维度
        reward = torch.tensor(unpacked[7]).float().detach().to(device)  # 转为张量
        next_UAV_obs = torch.stack(unpacked[8]).float().to(device)  # 堆叠张量
        done = torch.tensor(unpacked[9]).float().detach().to(device)  # 转为张量
        selected_UAV = torch.stack(unpacked[10]).long().detach().to(device)  # 转为张量并扩展维度

        return (
            samp_number,  # 返回编号
            x_tensor,  # 返回 x_tensor
            y_tensor,  # 返回 y_tensor
            d_tensor,  # 返回 d_tensor
            UAV_obs,  # 返回 UAV_obs
            maintenance_action,  # 返回 maintenance_action
            routing_action,  # 返回 routing_action
            reward,  # 返回 reward
            next_UAV_obs,  # 返回 next_UAV_obs
            done,  # 返回 done
            selected_UAV  # 返回 selected_UAV
        )

    def size(self):  # 获取 buffer 中数据的数量
        return len(self.buffer)



# 定义MADDPG算法
class MADDPG:
    def __init__(self, obs_dim, maintenance_action_dim, routing_action_dim, num_customers, num_UAVs, lr=5e-5, gamma=0.95, \
                 buffer_capacity=5000, batch_size=256, tau=0.005):
        self.obs_dim = obs_dim
        self.maintenance_action_dim = maintenance_action_dim
        self.routing_action_dim = routing_action_dim
        self.num_customers = num_customers
        self.num_UAVs = num_UAVs
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # 软更新的tau系数

        # 共享的actor和critic网络
        self.actor = ActorNetwork(obs_dim, num_customers).to(device)
        self.critic = CriticNetwork(obs_dim, maintenance_action_dim, routing_action_dim, selected_UAV_dim, num_customers).to(device)

        # 目标网络
        self.target_actor = ActorNetwork(obs_dim, num_customers).to(device)
        self.target_critic = CriticNetwork(obs_dim, maintenance_action_dim, routing_action_dim, selected_UAV_dim, num_customers).to(device)

        # 将目标网络的参数初始化为在线网络的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        #优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def select_action(self, x_tensor, y_tensor, d_tensor, UAV_obs_tensor_i, selected_UAV):
        
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
        d_tensor = d_tensor.to(device)
        
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)
        
        # 将观测和mask输入actor，选择动作
        maintenance_action, routing_probs = self.actor(x_tensor, y_tensor, d_tensor, UAV_obs_tensor_i, selected_UAV)
        
        # 从routing_probs中选择概率最大的那个点，这里本来是size[7],argmax去掉一个维度后，u(0)增加一个维度，最后变为1维张量
        routing_action = torch.argmax(routing_probs, dim=-1).unsqueeze(0).to(device)

        # 获取状态
        UAV = torch.tensor(env.UAV_obs_matrix, dtype=torch.float32).to(device)
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

    
    def update(self, samp_number_tensor, x_tensor, y_tensor, d_tensor, UAV_obs, maintenance_action, routing_action, rewards, next_UAV_obs, done, selected_UAV):
        
        transition_dict = {
            'samp_number_tensor': samp_number_tensor,
            'x_tensor': x_tensor,
            'y_tensor': y_tensor,
            'd_tensor': d_tensor,
            'UAV_obs': UAV_obs,
            'maintenance_action': maintenance_action,
            'routing_action': routing_action,
            'rewards': rewards,
            'next_UAV_obs': next_UAV_obs,
            'done': done,
            'selected_UAV': selected_UAV
        }

        # 将 UAV_obs 维度调整为 2D
        samp_number_tensor = transition_dict['samp_number_tensor'].clone().detach().to(device).float()
        x_tensor = transition_dict['x_tensor'].clone().detach().to(device).float()
        y_tensor = transition_dict['y_tensor'].clone().detach().to(device).float()
        d_tensor = transition_dict['d_tensor'].clone().detach().to(device).float()

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
            selected_UAV = torch.tensor(selected_UAV, dtype=torch.float32).unsqueeze(0).to(device)


        # Critic更新：TD误差更新Q值
        q_value = self.critic(x_tensor, y_tensor, d_tensor, UAV_obs, maintenance_action, routing_action, selected_UAV).to(device)

        # 获取下一步的maintenance_action和routing_action
        with torch.no_grad():
            
            # 生成下一步的 next_selected_UAV
            broken_check = next_UAV_obs[:, :, 6] #形状[256,num_UAVs]
            has_1 = (broken_check == 1) #检查每个样本的第7列是否有1，形状[256,num_UAVs]

            # 如果存在 broken == 1，找到第一个出现的位置
            if has_1.any():
                broken_indices = torch.argmax(has_1.int(), dim=1)  # 转换为整型后使用 argmax
            else:
                # 定义一个默认值，避免未定义 broken_indices
                broken_indices = torch.full((next_UAV_obs.size(0),), -1, dtype=torch.long, device=next_UAV_obs.device)

            # 找到第8列的最小值索引
            decision_time_check = next_UAV_obs[:, :, 7]
            min_indices = torch.argmin(decision_time_check, dim=1)  # 形状 [256]

            # 最终选择
            next_selected_UAV = torch.where(has_1.any(dim=1), broken_indices, min_indices)  # [256]
            next_selected_UAV = next_selected_UAV.unsqueeze(1)  # 调整为 [256, 1]
            
            next_selected_UAV = next_selected_UAV.to(device)

            # 对采样的batchsize个next_UAV_obs进行selected_UAV提取对应无人机的观测
            next_UAV_obs_next_i = next_UAV_obs[torch.arange(self.batch_size), next_selected_UAV.squeeze(), :].to(device)

            next_maintenance_action, next_routing_probs = self.actor(x_tensor, y_tensor, d_tensor, next_UAV_obs_next_i, next_selected_UAV)
            next_routing_action = torch.argmax(next_routing_probs, dim=-1).unsqueeze(1).to(device)
            # torch.argmax 返回的size是[batch size]，u(1)会在索引1的位置添加一个新的维度，将张量的形状从[256]变为[256, 1]
            # 这里1 和 -1均可，在索引位置 -1 上添加一个新的维度，即在最后一个维度上添加一个维度
            
            next_q_value = self.critic(x_tensor, y_tensor, d_tensor, next_UAV_obs, next_maintenance_action, \
                next_routing_action, next_selected_UAV).to(device)
            
            rewards = rewards.view(-1, 1).to(device)  # 将 rewards 从 [256] 调整为 [256, 1]
            done = done.view(-1, 1).to(device)        # 将 done 从 [256] 调整为 [256, 1]

            # TD目标：reward + 折扣因子 * 下一步的Q值 * (1 - done)
            target_q_value = (rewards + self.gamma * (1 - done) * next_q_value).to(device)
        
        # 计算critic损失并更新critic
        critic_loss = nn.MSELoss()(q_value, target_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新：最大化Q值
        maintenance_action, routing_probs = self.actor(x_tensor, y_tensor, d_tensor, UAV_obs[torch.arange(self.batch_size), selected_UAV.long().squeeze(), :], selected_UAV)
        routing_action = torch.argmax(routing_probs, dim=-1).unsqueeze(1).to(device)
        
        # 计算actor损失，使用负Q值的平均值
        actor_loss = -self.critic(x_tensor, y_tensor, d_tensor, UAV_obs, maintenance_action, routing_action, selected_UAV).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 执行软更新（Soft Update）
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, online_network, target_network):
        """
        执行软更新，将在线网络（online_network）的参数按照tau比例更新到目标网络（target_network）。
        """
        for online_params, target_params in zip(online_network.parameters(), target_network.parameters()):
            target_params.data.copy_(
                target_params.data * (1.0 - self.tau) + online_params.data * self.tau
            )


# 2. PPOActor 网络
# ==========================
class PPOActor(nn.Module):
    def __init__(self, UAV_input_dim, num_UAVs, num_customers, hidden_dim=128):
        """
        :param state_dim: 状态向量的总维度
        :param num_UAVs: UAV的数量
        :param num_customers: 客户点的数量
        :param hidden_dim: 隐藏层神经元数量
        """
        super(PPOActor, self).__init__()
        self.num_UAVs = num_UAVs
        self.num_customers = num_customers
        
        # 全连接层1
        self.fc1 = nn.Linear(num_UAVs * UAV_input_dim + 3, hidden_dim)
        # 全连接层2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层：为 (num_customers * num_UAVs) 个 logits
        self.fc3 = nn.Linear(hidden_dim, num_UAVs)
        
    def forward(self, x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor):
        """
        前向传播:
        :param x: [batch_size, state_dim]
        :return: shape=[batch_size, num_customers, num_UAVs] 的 logits
        """

        # 将输入的 x, y, d, UAV_obs_tensor 拼接起来
        x_i_tensor = x_i_tensor.to(device)
        y_i_tensor = y_i_tensor.to(device)
        d_i_tensor = d_i_tensor.to(device)
        UAV_obs_tensor = UAV_obs_tensor.to(device)
        UAV_obs_tensor = UAV_obs_tensor.view(-1,1).squeeze(1)

        x = torch.cat([x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor], dim=-1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)  # shape: [batch_size, num_customers * num_UAVs]
        
        # 按照 UAV 维度做 softmax，得到每个顾客点分给 UAV 的概率
        logits = torch.softmax(logits, dim=-1)  # softmax over UAVs for each customer

        return logits  # [batch_size, num_UAVs, num_customers]

# ==========================
# 3. PPOCritic 网络
# ==========================
class PPOCritic(nn.Module):
    def __init__(self, UAV_input_dim, hidden_dim=128):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(UAV_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层：一个标量，用于表示V(s)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor):
        """
        前向传播:
        :param x: [batch_size, state_dim]
        :return: [batch_size, 1]
        """
        x_i_tensor = x_i_tensor.to(device)
        y_i_tensor = y_i_tensor.to(device)
        d_i_tensor = d_i_tensor.to(device)
        UAV_obs_tensor = UAV_obs_tensor.to(device)
        UAV_obs_tensor = UAV_obs_tensor.view(x_i_tensor.shape[0], -1)

        x = torch.cat([x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# ==========================
# 4. 经验回放结构
# ==========================
class RolloutBuffer:
    """
    用于存储一次或多次交互后积累的数据，供PPO训练
    """
    def __init__(self):
        self.UAV_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        del self.UAV_obs[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

# ==========================
# 5. PPO 主体
# ==========================
class PPO:
    def __init__(self,
                 UAV_input_dim,
                 num_UAVs,
                 num_customers,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 gamma=0.99,
                 k_epochs=10,
                 eps_clip=0.2,
                 hidden_dim=128):
        """
        :param state_dim: 状态维度
        :param num_UAVs: UAV数量
        :param num_customers: 客户点数量
        :param lr_actor: Actor学习率
        :param lr_critic: Critic学习率
        :param gamma: 折扣因子
        :param k_epochs: 每次更新时的迭代轮数
        :param eps_clip: PPO裁剪范围
        :param hidden_dim: 隐藏层规模
        """
        self.UAV_input_dim = UAV_input_dim
        self.num_UAVs = num_UAVs
        self.num_customers = num_customers
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 建立 Actor 和 Critic
        self.actor = PPOActor(UAV_input_dim, num_UAVs, num_customers, hidden_dim).to(device)
        self.critic = PPOCritic(UAV_input_dim, hidden_dim).to(device)

        # 优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 经验回放缓冲
        self.buffer = RolloutBuffer()

    def select_action(self, x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor):
        """
        根据当前策略选取动作 (分配矩阵)
        
        :param state: numpy 数组 shape [state_dim, ]，单步
        :param customer_status: shape [num_customers, ]，表示每个客户点对本次分配的需求情况
               -1 表示“待分配”，否则表示“不用分配”，可以是0或1等标识
        :param use_argmax: True表示每个客户点选取概率最大的 UAV，False表示随机采样(适合训练)
        :return: 
          action_matrix: [num_customers, num_UAVs] 的0-1分配矩阵
          log_prob_sum: 该动作的log_prob之和(标量)
          value: Critic对当前状态的价值估计(标量，用于后续计算优势函数)
        """
        # 转为tensor，放到GPU
        x_i_tensor = x_i_tensor.to(device)
        y_i_tensor = y_i_tensor.to(device)
        d_i_tensor = d_i_tensor.to(device)
        
        UAV_obs_tensor = UAV_obs_tensor.to(device)
        
        # 得到logits => shape=[1, num_customers, num_UAVs]
        logits = self.actor(x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor)  # [1, C, U]


        m = torch.distributions.Categorical(logits)
        CA_actions = m.sample()  # [B, C]，每个客户点选中的 UAV 编号
        index=CA_actions.unsqueeze(0).unsqueeze(0)

        #计算log_prob_sum
        log_prob = m.log_prob(CA_actions)  # [B]
        log_prob_sum = log_prob.sum()  # [B]

        batch_size = x_i_tensor.shape[0]  # 获取批次大小
        assignment_matrix = torch.zeros(batch_size, num_UAVs, device=logits.device)  # 初始化分配矩阵 [B, U]
        assignment_matrix.scatter_(dim=1, index=index, value=1)  # 将选中的 UAV 编号位置置为 1

        return assignment_matrix, CA_actions, log_prob_sum


    def remember(self, UAV_obs, action_matrix, log_prob, reward, done, value):
        """
        存储交互数据到buffer
        :param state: 当前时刻状态(一维numpy)
        :param action_matrix: [num_uavs, num_customers]的0-1分配矩阵
        :param log_prob: 对应动作的对数概率(标量)
        :param reward: 环境反馈的即时奖励
        :param done: 是否回合结束
        :param value: Critic对当前状态的价值估计(标量)
        """
        self.buffer.UAV_obs.append(UAV_obs)
        # 动作扁平化后存储
        # 形状 [num_uavs * num_customers]
        action_flat = action_matrix.flatten()  
        self.buffer.actions.append(action_flat)
        self.buffer.log_probs.append(log_prob.item())  # 转为标量存储
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.buffer.values.append(value)

    def update(self):
        """
        使用回放缓冲中的数据进行 PPO 更新
        """
        # 1. 把经验回放的数据转换为 Tensor
        old_states = torch.FloatTensor(self.buffer.UAV_obs).to(device)
        old_actions = torch.FloatTensor(self.buffer.actions).to(device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(device)
        old_values = torch.FloatTensor(self.buffer.values).to(device)
        is_terminals = self.buffer.is_terminals

        # 2. 计算回合/序列的Returns和Advantage
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(is_terminals)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(device)

        advantages = returns - old_values  # 简单 Advantage

        # 3. PPO多轮更新
        batch_size = len(old_states)
        for _ in range(self.k_epochs):
            # (a) 计算新策略下的 log_prob、价值
            logits = self.actor(old_states)  # [batch_size, num_customers, num_UAVs]
            prob_matrix = torch.softmax(logits, dim=-1)  # [batch_size, num_customers, num_UAVs]

            # 因为我们在 buffer 里存的是 action 的 0-1 展开向量
            # 需要还原为 [batch_size, num_customers, num_UAVs]
            reshaped_actions = old_actions.view(batch_size, self.num_UAVs, self.num_customers)

            log_probs_new = []
            for i in range(batch_size):
                single_prob_list = []
                for c in range(self.num_customers):
                    # 找到分配的uav索引
                    uav_idx = (reshaped_actions[i, :, c] == 1).nonzero(as_tuple=True)
                    if len(uav_idx[0]) == 0:
                        # 如果没选任何UAV(极端情况)
                        p = 1e-8
                    else:
                        idx = uav_idx[0].item()
                        p = prob_matrix[i, idx, c]
                    single_prob_list.append(p)
                single_prob_tensor = torch.stack(single_prob_list)
                log_probs_new.append(torch.log(single_prob_tensor + 1e-8).sum())
            new_log_probs = torch.stack(log_probs_new)  # [batch_size]

            values = self.critic(old_states).squeeze(1)  # [batch_size]

            # (b) ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # (c) PPO 裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # (d) Critic损失
            critic_loss = nn.MSELoss()(values, returns)

            # (e) 反向传播更新
            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # 4. 清空缓冲区
        self.buffer.clear()

# ==========================









# 初始化ReplayBuffer
replay_buffer = ReplayBuffer(capacity=5000)  


# 初始化环境和算法

obs_dim = env.observation_space.shape[0]  # 观测维度
num_customers = env.num_customers  # 客户数量
num_UAVs = env.num_UAVs  # 无人机数量
maintenance_action_dim = 1  
routing_action_dim = 1  
selected_UAV_dim = 1
samp_number_dim = 1


maddpg = MADDPG(obs_dim, maintenance_action_dim, routing_action_dim, num_customers, num_UAVs = env.num_UAVs)

ppo = PPO(UAV_input_dim=8 + num_customers, num_UAVs=num_UAVs, num_customers=num_customers)



samp_number = 1  # 初始采样编号
max_samp_number = 1200  # 最大采样编号
# 初始化 recent_costs 用于记录最近的 episode_cost
recent_costs = []  # 存储最近 10 个 episode 的 cost

# 初始化每台无人机的路径和总reward存储
UAV_routes = [[] for _ in range(num_UAVs)]  # 存储每台无人机的路径
UAV_rewards = [0 for _ in range(num_UAVs)]  # 存储每台无人机的reward
total_reward = 0
episode_rewards = [] # 存储每个episode的reward

# 创建SummaryWriter对象来记录数据
log_dir = 'D:\\JNU\\AApaper\\code\\new\\up\\record'
writer = SummaryWriter(log_dir=log_dir)
#输入这个：tensorboard --logdir=D:\JNU\AApaper\code\somepic

# 指定日志文件路径
log_file_path = 'D:\\JNU\\AApaper\\code\\new\\up\\record\\1.txt'
# 新的网络参数日志路径（每50代保存一次）
network_log_file_path = 'D:\\JNU\\AApaper\\code\\new\\up\\record\\2.txt'
seed_log_file_path = 'D:\\JNU\\AApaper\\code\\new\\up\\record\\3.txt'

# 如果文件已存在，清空文件内容
if os.path.exists(log_file_path):
    open(log_file_path, 'w').close()
if os.path.exists(network_log_file_path):
    open(network_log_file_path, 'w').close()
if os.path.exists(seed_log_file_path):
    open(seed_log_file_path, 'w').close()

# 添加更新记录变量
record_interval = 50  # 每 50 代记录一次参数
clear_interval = 1   # 每记录 1 次清空文件
record_count = 0  # 用于记录已保存的次数


# 添加更新间隔变量
update_interval = 50
step_counter = 0

# 开始训练
# train_maddpg(env, maddpg)
num_episodes = 50
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

        # 获取obs
        UAV_obs_array = np.array(UAV_obs)
        UAV_obs_tensor = torch.tensor(UAV_obs_array, dtype=torch.float32).to(device)
        
        # 获取倒数客户列
        task = UAV_obs_tensor[:, -num_customers:]

        # 判断决策智能体是否为中央控制智能体（Y：customer_state_space全0 + 出现-1 // 分别表示初始分配+故障重分配）
        # 检查是否全为0
        # 对每一个客户点检查是否还没有被分配（某一列全为 0）
        for i in range(num_customers):
            if torch.all(task[:, i] == 0):  # 第 i 个客户点还没有被分配
        # 获取客户点 i 的信息
                x_i = x_tensor[0]
                y_i = y_tensor[0]
                d_i = d_tensor[0]   
        
                #把xyd转换为tensor
                x_i_tensor = torch.tensor(x_i, dtype=torch.float32).to(device).unsqueeze(0)
                y_i_tensor = torch.tensor(y_i, dtype=torch.float32).to(device).unsqueeze(0)
                d_i_tensor = torch.tensor(d_i, dtype=torch.float32).to(device).unsqueeze(0)
        
                # 执行 PPO 动作选择
                assignment_action, _, _= ppo.select_action(x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor)

                selected_customer = i  # 记录被选中的客户点编号
                selected_customer_tensor = torch.tensor(selected_customer, dtype=torch.float32).unsqueeze(0).to(device)
                # 执行一步动作
                next_UAV_obs, reward_CA, done_CA, cost_CA = env.step_CA(assignment_action, selected_customer_tensor)
                #reward_CA_tensor = torch.tensor(reward_CA, dtype=torch.float32).to(device)
                #done_CA_tensor = torch.tensor(done_CA, dtype=torch.float32).to(device)

                # 把next_UAV_obs转换为tensor
                next_UAV_obs_array = np.array(next_UAV_obs)
                next_UAV_obs_tensor = torch.tensor(next_UAV_obs_array, dtype=torch.float32).to(device)
                UAV_obs = next_UAV_obs
        

        # 检查倒数6列中是否有 -1 值，如果有 -1 值，才进行后续处理
        for i in range(num_customers):
            if torch.any(task[:, i] == -1):  # 如果这一列中有 -1 值，表示需要重新分配
                x_i = x_tensor[0]
                y_i = y_tensor[0]
                d_i = d_tensor[0]
        
                #把xyd转换为tensor
                x_i_tensor = torch.tensor(x_i, dtype=torch.float32).to(device).unsqueeze(0)
                y_i_tensor = torch.tensor(y_i, dtype=torch.float32).to(device).unsqueeze(0)
                d_i_tensor = torch.tensor(d_i, dtype=torch.float32).to(device).unsqueeze(0)
        
                # 执行 PPO 动作选择
                assignment_action, _, _= ppo.select_action(x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor)

                selected_customer = i  # 记录被选中的客户点编号
                selected_customer_tensor = torch.tensor(selected_customer, dtype=torch.float32).unsqueeze(0).to(device)
                # 执行一步动作
                next_UAV_obs, reward_CA, done_CA, cost_CA = env.step_CA(assignment_action, selected_customer_tensor)
                #reward_CA_tensor = torch.tensor(reward_CA, dtype=torch.float32).to(device)
                #done_CA_tensor = torch.tensor(done_CA, dtype=torch.float32).to(device)

                # 把next_UAV_obs转换为tensor
                next_UAV_obs_array = np.array(next_UAV_obs)
                next_UAV_obs_tensor = torch.tensor(next_UAV_obs_array, dtype=torch.float32).to(device)
                UAV_obs = next_UAV_obs

        # 在此处可以对 assignment_action 进行处理

        # PPO分配：循环集合中的点，逐个分配，直至空集

            # 执行分配决策更新状态




        


        selected_UAV = env.selected_UAV()
        #把selected_UAV转换为tensor
        selected_UAV_tensor = torch.tensor(selected_UAV, dtype=torch.float32).unsqueeze(0).to(device)
        # 将采样编号变成张量
        samp_number_tensor = torch.tensor(samp_number, dtype=torch.float32).unsqueeze(0).to(device)
        

        # 获取obs
        UAV_obs_array = np.array(UAV_obs)
        UAV_obs_tensor = torch.tensor(UAV_obs_array, dtype=torch.float32).to(device)
        UAV_obs_tensor_i = UAV_obs_tensor[int(selected_UAV_tensor.item())].clone().to(device)
           

        # 根据obs选择动作
        maintenance_action, routing_action = maddpg.select_action(x_tensor, y_tensor, d_tensor, UAV_obs_tensor_i, selected_UAV)

        # 执行一步动作
        next_UAV_obs, rewards, done, cost = env.step(maintenance_action, routing_action , selected_UAV)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

        # 把next_UAV_obs转换为tensor
        next_UAV_obs_array = np.array(next_UAV_obs)
        next_UAV_obs_tensor = torch.tensor(next_UAV_obs_array, dtype=torch.float32).to(device)


        # 将experience加入到ReplayBuffer
        replay_buffer.add(samp_number_tensor, x_tensor, y_tensor, d_tensor, UAV_obs_tensor, maintenance_action, routing_action, \
                          rewards_tensor, next_UAV_obs_tensor, done_tensor, selected_UAV_tensor)
            

        # 每隔固定步数（50步）更新MADDPG
        if step_counter % update_interval == 0 and replay_buffer.size() >= maddpg.batch_size:  # 确保buffer中有足够的数据
            batch = replay_buffer.sample(maddpg.batch_size)
            samp_number_tensor_batch, x_tensor_batch, y_tensor_batch, d_tensor_batch, UAV_obs_batch, maintenance_action_batch, routing_action_batch, \
                rewards_batch, next_UAV_obs_batch, done_batch, selected_UAV_batch = batch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = tuple(t.to(device) for t in batch)  # 将所有批量数据转移到设备

            transition_dict = {
                'samp_number_tensor': batch[0],
                'x_tensor': batch[1],
                'y_tensor': batch[2],
                'd_tensor': batch[3],
                'UAV_obs': batch[4],
                'maintenance_action': batch[5],
                'routing_action': batch[6],
                'rewards': batch[7],
                'next_UAV_obs': batch[8],
                'done': batch[9],
                'selected_UAV': batch[10]
            }
                            
            critic_loss, actor_loss = maddpg.update(**transition_dict)

            writer.add_scalar('NO3_Losses/Critic Loss', critic_loss, step_counter)  # 记录critic_loss
            writer.add_scalar('NO3_Losses/Actor Loss', actor_loss, step_counter)  # 记录actor_loss

            # 累加损失值到本episode
            episode_critic_loss += critic_loss  # 累加critic损失
            episode_actor_loss += actor_loss    # 累加actor损失            

        # 打印当前buffer大小和其中的数据
        print(f"Buffer size: {replay_buffer.size()}")
        """ for experience in replay_buffer.buffer:
            print(f"Stored Experience: {experience}")
        """
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
    writer.add_scalar('NO4_Episode Losses/Episode Critic Loss', episode_critic_loss, episode)
    writer.add_scalar('NO4_Episode Losses/Episode Actor Loss', episode_actor_loss, episode) 

    print(f"Episode {episode} finished. Total Reward: {episode_reward}, Critic Loss: {episode_critic_loss}, Actor Loss: {episode_actor_loss}, Cost:{episode_cost.item():.4f}")

    #episode_rewards用来这里画图的（其实可以不需要了，因为上面已经记录了episode_reward用tensorboard画了）
    episode_rewards.append(total_reward)
    print(f'Episode {episode} completed. Total reward: {total_reward}')

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
            for i, param in enumerate(maddpg.actor.parameters()):
                f.write(f"Actor Param {i}: {param.data.tolist()}\n")
            for i, param in enumerate(maddpg.critic.parameters()):
                f.write(f"Critic Param {i}: {param.data.tolist()}\n")

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

    """
    # 检查是否所有批次都已经收敛
    if samp_number > max_samp_number:  # 如果已经完成所有批次客户点训练
        print("所有批次的客户点已经收敛，提前停止训练。")
        done = True  # 结束训练
    """

# 训练结束后关闭TensorBoard writer
writer.close()

# 最终打印每台无人机的完整飞行路线和总reward
for i in range(num_UAVs):
    print(f'UAV{i + 1} routes: {"-".join(map(str, UAV_routes[i]))}, UAV{i + 1} reward: {UAV_rewards[i]}')

print(f'Total reward: {total_reward}')
print(f'Episode rewards: {episode_rewards}')

# 绘制奖励图
plt.plot(range(1, num_episodes + 1), episode_rewards, label="Total Reward per Episode")  # 确保添加 label
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.legend()  # 确保调用 legend() 以显示图例
plt.show()

# 假设 self.actor 是你的 Actor 网络实例
save_dir = 'D:\\JNU\\AApaper\\code\\new\\up\\record'  # 保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 如果目录不存在，则创建

save_path = os.path.join(save_dir, 'actor_network_full.pth')  # 完整文件路径

# 保存整个模型
torch.save(maddpg.actor, save_path)
print(f"模型已保存到：{save_path}")