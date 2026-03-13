"""
PPO+MADDPG 联合训练脚本。
- MADDPG 处理路径规划与维修决策（从 maddpg.py 导入，加载预训练权重）
- PPO 处理客户-UAV 分配（用 GA 数据预训练，替代原 KMeans）
- 训练流程：冻结 MADDPG → 训练 PPO 至收敛 → 交替训练（基于收敛切换）
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from environment_up import Environment
import maddpg as maddpg_module
from maddpg import ReplayBuffer, get_device

env = Environment(num_UAVs=3, num_customers=10, quiet=True)
# 动态客户参数可在此覆盖环境默认值（默认 prob=0.1, max_new=2）
# env.config['dynamic_customer_prob'] = 0.05
# env.config['dynamic_max_new_per_step'] = 1
maddpg_module._default_env = env

device = get_device()
print(f"[PPO+MADDPG] Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

num_customers = env.num_customers
num_UAVs = env.num_UAVs


# ==========================
# PPOActor 网络
# ==========================
class PPOActor(nn.Module):
    def __init__(self, state_dim, num_UAVs, hidden_dim=128):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_UAVs)

    def forward(self, state):
        x = state.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# ==========================
# PPOCritic 网络
# ==========================
class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = state.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# ==========================
# PPO RolloutBuffer
# ==========================
class RolloutBuffer:
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
# PPO 算法
# ==========================
class PPO:
    def __init__(self, UAV_input_dim, num_UAVs, num_customers,
                 lr_actor=2e-5, lr_critic=2e-5, gamma=0.99,
                 k_epochs=4, eps_clip=0.15, hidden_dim=128,
                 gae_lambda=0.95):
        self.UAV_input_dim = UAV_input_dim
        self.num_UAVs = num_UAVs
        self.num_customers = num_customers
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda

        self.actor = PPOActor(UAV_input_dim, num_UAVs, hidden_dim).to(device)
        self.critic = PPOCritic(UAV_input_dim, hidden_dim).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def _build_state(self, x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor):
        """全局观测: [当前客户 x,y,d | 所有UAV状态(8维×num_UAVs) | 所有UAV位置/目的地坐标 | 完整任务分配矩阵(num_UAVs×num_customers)]"""
        x_i_tensor = x_i_tensor.to(device)
        y_i_tensor = y_i_tensor.to(device)
        d_i_tensor = d_i_tensor.to(device)
        UAV_obs_tensor = UAV_obs_tensor.to(device)

        customer_info = torch.stack(
            [x_i_tensor.squeeze(), y_i_tensor.squeeze(), d_i_tensor.squeeze()],
            dim=-1
        ).unsqueeze(0)  # [1, 3]

        # 原始 UAV 状态与任务矩阵
        uav_states = UAV_obs_tensor[:, :8]  # [num_UAVs, 8]
        task_matrix = UAV_obs_tensor[:, -self.num_customers:]  # [num_UAVs, num_customers]

        # 将 UAV 的位置/目的地节点 id 映射为实际坐标 (0 = 仓库(0,0), 1..N = 客户坐标)
        position_ids = uav_states[:, 0].long()
        destination_ids = uav_states[:, 1].long()

        num_uavs = self.num_UAVs
        uav_pos_coords = torch.zeros((num_uavs, 2), dtype=torch.float32, device=device)
        uav_dest_coords = torch.zeros((num_uavs, 2), dtype=torch.float32, device=device)

        # 使用环境中的最新客户坐标进行映射（支持动态客户）
        customer_positions_np = env.customer_positions  # [num_customers, 2]
        customer_positions = torch.as_tensor(customer_positions_np, dtype=torch.float32, device=device)

        pos_mask = position_ids > 0
        if pos_mask.any():
            uav_pos_coords[pos_mask] = customer_positions[position_ids[pos_mask] - 1]

        dest_mask = destination_ids > 0
        if dest_mask.any():
            uav_dest_coords[dest_mask] = customer_positions[destination_ids[dest_mask] - 1]

        uav_states_flat = uav_states.reshape(1, -1)                # [1, num_UAVs * 8]
        uav_pos_coords_flat = uav_pos_coords.reshape(1, -1)        # [1, num_UAVs * 2]
        uav_dest_coords_flat = uav_dest_coords.reshape(1, -1)      # [1, num_UAVs * 2]
        task_matrix_flat = task_matrix.reshape(1, -1)              # [1, num_UAVs * num_customers]

        state = torch.cat(
            [customer_info, uav_states_flat, uav_pos_coords_flat, uav_dest_coords_flat, task_matrix_flat],
            dim=-1
        )
        return state

    @torch.no_grad()
    def select_action(self, x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor,
                      deterministic=False):
        UAV_obs_tensor = UAV_obs_tensor.to(device)
        state = self._build_state(x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor)

        logits = self.actor(state)

        # 基于 broken 标志对损坏 UAV 做动作屏蔽
        broken_flags = UAV_obs_tensor[:, 6]  # 1 = 损坏, 0 = 正常
        broken_mask = broken_flags >= 0.5

        if broken_mask.any() and not broken_mask.all():
            masked_logits = logits.clone()
            masked_logits[0, broken_mask] = -1e9
        else:
            # 若全部正常或全部损坏，则暂不强制屏蔽，保持 logits 原样
            masked_logits = logits

        probs = torch.softmax(masked_logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.zeros_like(action, dtype=torch.float32)
        else:
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

        value = self.critic(state).squeeze(-1)

        assignment_one_hot = torch.zeros_like(probs)
        assignment_one_hot.scatter_(1, action.unsqueeze(-1), 1.0)

        return (
            assignment_one_hot.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
            value.squeeze(0),
            state.squeeze(0)
        )

    def remember(self, state, action, log_prob, reward, done, value):
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = np.array(state, dtype=np.float32)
        self.buffer.UAV_obs.append(state_np)

        if isinstance(action, torch.Tensor):
            action_val = int(action.item())
        else:
            action_val = int(action)
        self.buffer.actions.append(action_val)

        if isinstance(log_prob, torch.Tensor):
            log_prob_val = float(log_prob.item())
        else:
            log_prob_val = float(log_prob)
        self.buffer.log_probs.append(log_prob_val)

        self.buffer.rewards.append(float(reward))
        self.buffer.is_terminals.append(bool(done))
        if isinstance(value, torch.Tensor):
            value_val = float(value.item())
        else:
            value_val = float(value)
        self.buffer.values.append(value_val)

    def update(self):
        if len(self.buffer.rewards) == 0:
            return None
        states_np = np.array(self.buffer.UAV_obs, dtype=np.float32)
        old_states = torch.from_numpy(states_np).to(device)
        old_actions = torch.LongTensor(self.buffer.actions).to(device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(device)
        old_values = torch.FloatTensor(self.buffer.values).to(device)
        is_terminals = self.buffer.is_terminals

        # GAE(λ) 优势估计
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=device)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if is_terminals[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * mask
            advantages[t] = gae
            next_value = old_values[t]

        # 对优势函数做标准化，有助于稳定 PPO 训练
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        returns = advantages + old_values

        # 统计信息累计（按 epoch 平均）
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0

        for _ in range(self.k_epochs):
            logits = self.actor(old_states)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(old_actions)

            values = self.critic(old_states).squeeze(1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # PPO-Clip 基础损失 + 熵正则化（系数 0.01，熵惩罚即减去熵以鼓励探索）
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            entropy_coef = 0.01
            actor_loss = policy_loss - entropy_coef * entropy

            critic_loss = nn.MSELoss()(values, returns)

            # 统计量
            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                clip_frac = (torch.abs(ratio - 1.0) > self.eps_clip).float().mean().item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_kl += approx_kl
                total_entropy += entropy.item()
                total_clip_frac += clip_frac

            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.optimizer_critic.step()

        self.buffer.clear()
        k = float(self.k_epochs)
        return {
            "actor_loss": total_actor_loss / k,
            "critic_loss": total_critic_loss / k,
            "approx_kl": total_kl / k,
            "entropy": total_entropy / k,
            "clip_fraction": total_clip_frac / k,
        }


# ==========================
# GA 数据预训练 PPO（替代原 KMeans）
# ==========================
def pretrain_ppo_with_ga_data(the_env, ppo, ga_data_path, num_iters=300):
    """
    读取 maddpg 阶段 GA 生成的客户分配数据 (JSONL)，对 PPO Actor 做监督预训练。
    每条记录包含一轮客户点的位置、需求及 GA 得出的最优分配（per_customer_assigned_uav）。
    """
    if not os.path.exists(ga_data_path):
        print(f"[PPO 预训练] GA 数据文件不存在: {ga_data_path}，跳过预训练")
        return

    records = []
    with open(ga_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        print("[PPO 预训练] GA 数据为空，跳过预训练")
        return

    n_uavs = ppo.num_UAVs
    n_cust = ppo.num_customers

    initial_uav_state = torch.zeros(8, dtype=torch.float32, device=device)
    initial_uav_state[4] = 100.0  # energy = 100
    uav_states_flat = initial_uav_state.repeat(n_uavs).unsqueeze(0)  # [1, num_UAVs*8]
    # 预训练阶段为 UAV 位置/目的地坐标提供占位零向量（与正式训练输入维度对齐）
    uav_pos_coords_flat = torch.zeros(1, n_uavs * 2, dtype=torch.float32, device=device)
    uav_dest_coords_flat = torch.zeros(1, n_uavs * 2, dtype=torch.float32, device=device)
    task_matrix_flat = torch.zeros(1, n_uavs * n_cust, dtype=torch.float32, device=device)

    states_all, targets_all = [], []
    for rec in records:
        positions = np.array(rec['customer_positions'])
        demands = np.array(rec['customer_demands'])
        assignment = rec['per_customer_assigned_uav']
        n_c = rec['num_customers']

        for j in range(n_c):
            target_uav = assignment[str(j)]
            if target_uav < 0:
                continue

            customer_info = torch.tensor(
                [positions[j][0], positions[j][1], demands[j]],
                dtype=torch.float32, device=device
            ).unsqueeze(0)  # [1, 3]

            state = torch.cat(
                [customer_info, uav_states_flat, uav_pos_coords_flat, uav_dest_coords_flat, task_matrix_flat],
                dim=-1
            )
            states_all.append(state.squeeze(0))
            targets_all.append(target_uav)

    if not states_all:
        print("[PPO 预训练] 无有效预训练样本，跳过")
        return

    states_t = torch.stack(states_all)
    targets_t = torch.tensor(targets_all, dtype=torch.long, device=device)

    print(f"[PPO 预训练] 使用 GA 数据，样本数: {len(states_all)}（来自 {len(records)} 轮客户点），迭代: {num_iters}")
    for it in range(num_iters):
        logits = ppo.actor(states_t)
        loss = F.cross_entropy(logits, targets_t)

        ppo.optimizer_actor.zero_grad()
        loss.backward()
        ppo.optimizer_actor.step()

        if (it + 1) % max(1, num_iters // 4) == 0:
            print(f"  迭代 {it + 1}/{num_iters}, loss: {loss.item():.4f}")

    print("[PPO 预训练] 完成。")


# ==========================
# 冻结 / 解冻工具
# ==========================
def freeze_params(*models):
    for m in models:
        for p in m.parameters():
            p.requires_grad = False


def unfreeze_params(*models):
    for m in models:
        for p in m.parameters():
            p.requires_grad = True


# ==========================
# 初始化
# ==========================
obs_dim = env.observation_space.shape[0]
maintenance_action_dim = 1
routing_action_dim = 1

# MADDPG: 从 maddpg.py 导入，传入 env
maddpg = maddpg_module.MADDPG(
    obs_dim, maintenance_action_dim, routing_action_dim,
    num_customers, num_UAVs, env=env,
    buffer_capacity=20000, batch_size=1024
)

# 加载 MADDPG 预训练权重
base_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(base_dir, "maddpgResult", "maddpg_pretrained.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    maddpg.actor.load_state_dict(ckpt['actor_state_dict'])
    maddpg.critic.load_state_dict(ckpt['critic_state_dict'])
    maddpg.target_actor.load_state_dict(ckpt['target_actor_state_dict'])
    maddpg.target_critic.load_state_dict(ckpt['target_critic_state_dict'])
    print(f"[MADDPG] 已加载预训练权重: {ckpt_path}")
else:
    print(f"[MADDPG] 预训练权重未找到 ({ckpt_path})，使用随机初始化")

# PPO（全局观测维度: 3 + num_UAVs*(8 + 2 + 2) + num_UAVs*num_customers）
# 8 = 原始 UAV 状态维度；2+2 = 当前位置(x,y)与目的地(x,y)坐标
ppo_state_dim = 3 + num_UAVs * (8 + 2 + 2) + num_UAVs * num_customers
ppo = PPO(UAV_input_dim=ppo_state_dim, num_UAVs=num_UAVs, num_customers=num_customers)

# PPO 用 GA 数据预训练
ga_data_path = os.path.join(base_dir, "maddpgResult", "ga_assignment_for_ppo.jsonl")
pretrain_ppo_with_ga_data(env, ppo, ga_data_path, num_iters=300)

# MADDPG 共享 ReplayBuffer（使用新版预分配张量+GMM 分层采样）
replay_buffer = ReplayBuffer(capacity=20000)

# Phase 1：冻结 MADDPG，先训练 PPO（target 网络不参与 freeze/unfreeze，只通过 soft_update 更新）
freeze_params(maddpg.actor, maddpg.critic)
training_who = "ppo"
print(f"[训练] Phase 1: 冻结 MADDPG，训练 PPO（动态客户按环境默认配置运行）")

# ==========================
# 训练配置
# ==========================
samp_number = 1
update_interval = 20
num_updates_per_interval = 3
step_counter = 0

# PPO 每次更新前累计的最小时间步数（按 remember 次数计）
ppo_update_every_steps = 50
warmup_episodes_after_switch = 30  # 切换后预热期：只收集数据不更新

max_episodes = 10000
min_episodes_before_switch = 100
_cv_schedule     = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08]
_window_schedule = [  30,   35,   40,   45,   50,   50]
recent_costs = []
phase_start_episode = 0
num_alternations = 0
warmup_remaining = 0  # 切换后预热倒计时

max_full_cycles_per_seed = 2  # 每批客户点最多完成几个完整 PPO-MADDPG 交替周期后轮换
full_cycle_count = 0  # 当前客户点上已完成的完整周期数
num_seed_rotations = 0  # 已完成的客户点轮换次数
max_seed_rotations = 30  # 最大客户轮换次数

epsilon_when_training = 0.10
epsilon_when_frozen = 0.0

# 日志路径
result_dir = os.path.join(base_dir, "result")
os.makedirs(result_dir, exist_ok=True)
log_dir = result_dir
writer = SummaryWriter(log_dir=log_dir)

log_file_path = os.path.join(result_dir, "training_log.txt")
seed_log_file_path = os.path.join(result_dir, "convergence_log.txt")
ppo_assign_log_path = os.path.join(result_dir, "ppo_assignment_log.txt")
for p in [log_file_path, seed_log_file_path, ppo_assign_log_path]:
    if os.path.exists(p):
        open(p, 'w').close()

train_start_time = time.time()
print(f"[训练] max_episodes={max_episodes}, MADDPG batch_size={maddpg.batch_size}, "
      f"update_interval={update_interval}")

# ==========================
# 训练循环
# ==========================
for episode in range(max_episodes):
    env.current_episode = episode
    env.current_step = 0

    train_ppo = (training_who == "ppo")
    train_maddpg = (training_who == "maddpg")
    current_epsilon = epsilon_when_training if train_maddpg else epsilon_when_frozen

    ppo_deterministic = not train_ppo

    UAV_rewards = [0.0 for _ in range(num_UAVs)]
    UAV_routes = [[] for _ in range(num_UAVs)]
    UAV_costs = [0.0 for _ in range(num_UAVs)]
    episode_reward = 0.0
    episode_cost = 0.0
    episode_critic_loss = 0.0
    episode_actor_loss = 0.0
    episode_steps = 0
    UAV_obs = env.reset()
    done = False
    ppo_assign_records = []
    initial_assignment_done = False  # 用于区分初始分配 vs 动态新客户

    while not done:
        x, y, d = env.customer_list()
        x_tensor = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).squeeze()
        y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device).squeeze()
        d_tensor = torch.as_tensor(np.asarray(d, dtype=np.float32), device=device).squeeze()

        UAV_obs_tensor = torch.as_tensor(np.array(UAV_obs), dtype=torch.float32, device=device)

        task = UAV_obs_tensor[:, -num_customers:]

        # PPO 分配：未分配的客户点（列全 0）或动态新客户（列含 -2）
        pending_customers = [i for i in range(num_customers) if bool(torch.all(task[:, i] == 0)) or bool(torch.any(task[:, i] == -2))]
        if pending_customers:
            for i in pending_customers:
                x_i_tensor = x_tensor[i].unsqueeze(0)
                y_i_tensor = y_tensor[i].unsqueeze(0)
                d_i_tensor = d_tensor[i].unsqueeze(0)

                assignment_action, uav_action, log_prob, value, ppo_state = ppo.select_action(
                    x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor,
                    deterministic=ppo_deterministic
                )

                selected_customer_tensor = torch.tensor(i, dtype=torch.float32).unsqueeze(0).to(device)
                next_UAV_obs, reward_CA, done_CA, cost_CA = env.step_CA(assignment_action, selected_customer_tensor)

                # 采用基础版奖励：当前所选 UAV 从当前位置到该客户点的旅行时间（越短越好）
                travel_time = float(cost_CA) / env.fix_v
                reward_ppo = -travel_time

                assigned_uav_idx = int(uav_action.item())
                is_new_customer = bool(torch.any(task[:, i] == -2))
                tag = "[动态新客户] " if is_new_customer else ""
                ppo_assign_records.append(
                    f"  {tag}客户{i} -> UAV{assigned_uav_idx}, T_travel={travel_time:.4f}, r={reward_ppo:.4f}"
                )

                if train_ppo:
                    ppo.remember(ppo_state, uav_action, log_prob,
                                 reward=reward_ppo, done=bool(done_CA), value=value)

                UAV_obs = next_UAV_obs
                UAV_obs_tensor = torch.as_tensor(np.array(UAV_obs), dtype=torch.float32, device=device)
                task = UAV_obs_tensor[:, -num_customers:]

        # PPO 故障重分配：UAV 故障导致的 -1 标记（某 UAV 的 1 -> -1）
        reassign_customers = [i for i in range(num_customers) if bool(torch.any(task[:, i] == -1))]
        if reassign_customers:
            for i in reassign_customers:
                x_i_tensor = x_tensor[i].unsqueeze(0)
                y_i_tensor = y_tensor[i].unsqueeze(0)
                d_i_tensor = d_tensor[i].unsqueeze(0)

                assignment_action, uav_action, log_prob, value, ppo_state = ppo.select_action(
                    x_i_tensor, y_i_tensor, d_i_tensor, UAV_obs_tensor,
                    deterministic=ppo_deterministic
                )

                selected_customer_tensor = torch.tensor(i, dtype=torch.float32).unsqueeze(0).to(device)
                next_UAV_obs, reward_CA, done_CA, cost_CA = env.step_CA(assignment_action, selected_customer_tensor)

                # 采用基础版奖励：当前所选 UAV 从当前位置到该客户点的旅行时间（越短越好）
                travel_time = float(cost_CA) / env.fix_v
                reward_ppo = -travel_time

                assigned_uav_idx = int(uav_action.item())
                ppo_assign_records.append(
                    f"  [重分配] 客户{i} -> UAV{assigned_uav_idx}, T_travel={travel_time:.4f}, r={reward_ppo:.4f}"
                )

                if train_ppo:
                    ppo.remember(ppo_state, uav_action, log_prob,
                                 reward=reward_ppo, done=bool(done_CA), value=value)

                UAV_obs = next_UAV_obs
                UAV_obs_tensor = torch.as_tensor(np.array(UAV_obs), dtype=torch.float32, device=device)
                task = UAV_obs_tensor[:, -num_customers:]

        # MADDPG 路径规划
        selected_UAV = env.selected_UAV()
        if selected_UAV < 0:
            break

        selected_UAV_tensor = torch.tensor(selected_UAV, dtype=torch.float32, device=device).unsqueeze(0)
        samp_number_tensor = torch.tensor(samp_number, dtype=torch.float32, device=device).unsqueeze(0)

        UAV_obs_tensor = torch.as_tensor(np.array(UAV_obs), dtype=torch.float32, device=device)
        UAV_obs_tensor_i = UAV_obs_tensor[selected_UAV].clone()

        maintenance_action, routing_action = maddpg.select_action(
            UAV_obs_tensor_i, selected_UAV, epsilon=current_epsilon
        )

        next_UAV_obs, rewards, done, cost = env.step(maintenance_action, routing_action, selected_UAV)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        done_tensor = torch.tensor(done, dtype=torch.float32, device=device)
        next_UAV_obs_tensor = torch.as_tensor(np.array(next_UAV_obs), dtype=torch.float32, device=device)

        ep_x_tensor = torch.tensor(env.customer_positions[:, 0], dtype=torch.float32, device=device)
        ep_y_tensor = torch.tensor(env.customer_positions[:, 1], dtype=torch.float32, device=device)
        ep_d_tensor = torch.tensor(env.customer_cargo_demands, dtype=torch.float32, device=device)

        replay_buffer.add(
            samp_number_tensor, ep_x_tensor, ep_y_tensor, ep_d_tensor,
            UAV_obs_tensor, maintenance_action, routing_action,
            rewards_tensor, next_UAV_obs_tensor, done_tensor, selected_UAV_tensor,
        )

        # MADDPG 更新（仅在训练 MADDPG 时，且预热期结束后）
        if (train_maddpg
                and warmup_remaining <= 0
                and step_counter % update_interval == 0
                and replay_buffer.size() >= maddpg.batch_size):
            for _ in range(num_updates_per_interval):
                batch = replay_buffer.sample(maddpg.batch_size)
                c_loss, a_loss = maddpg.update(
                    samp_number_tensor=batch[0], x_tensor=batch[1],
                    y_tensor=batch[2], d_tensor=batch[3],
                    UAV_obs=batch[4], maintenance_action=batch[5],
                    routing_action=batch[6], rewards=batch[7],
                    next_UAV_obs=batch[8], done=batch[9],
                    selected_UAV=batch[10],
                )
                episode_critic_loss += c_loss
                episode_actor_loss += a_loss
            writer.add_scalar('maddpg/step_critic_loss', c_loss, step_counter)
            writer.add_scalar('maddpg/step_actor_loss', a_loss, step_counter)

        dest = int(routing_action.item() if hasattr(routing_action, "item") else routing_action)
        if isinstance(selected_UAV, torch.Tensor):
            sel_int = int(selected_UAV.item())
        else:
            sel_int = int(selected_UAV)
        UAV_routes[sel_int].append(dest)
        UAV_rewards[sel_int] += float(rewards)
        UAV_costs[sel_int] += float(cost.item() if hasattr(cost, "item") else cost)

        UAV_obs = next_UAV_obs
        step_counter += 1
        episode_steps += 1
        initial_assignment_done = True  # 已做过至少一次路由，之后出现的状态0均为动态新客户

    total_reward = sum(UAV_rewards)
    total_cost = sum(UAV_costs)
    episode_reward = total_reward
    episode_cost = total_cost

    # 计算各 UAV 完成时间与 makespan（T_max = max_i T_i）
    uav_completion_times = [float(env.UAV_state["8_decision_time"][i, 0]) for i in range(num_UAVs)]
    T_max = max(uav_completion_times) if uav_completion_times else 0.0
    T_min = min(uav_completion_times) if uav_completion_times else 0.0

    # PPO：按累计 timestep 的更新
    if train_ppo and len(ppo.buffer.rewards) > 0:
        # 记录当前 PPO buffer 的整体统计信息（跨 episode）
        ppo_rewards_arr = np.array(ppo.buffer.rewards, dtype=np.float32)
        writer.add_scalar('ppo/buffer_mean_reward', float(ppo_rewards_arr.mean()), episode)
        writer.add_scalar('ppo/buffer_total_reward', float(ppo_rewards_arr.sum()), episode)
        writer.add_scalar('ppo/buffer_num_steps', int(ppo_rewards_arr.shape[0]), episode)

        # 按累计 timestep 触发 PPO 参数更新，并记录更新诊断信息
        if len(ppo.buffer.rewards) >= ppo_update_every_steps:
            update_info = ppo.update()
            if update_info is not None:
                writer.add_scalar('ppo/update_actor_loss', update_info["actor_loss"], episode)
                writer.add_scalar('ppo/update_critic_loss', update_info["critic_loss"], episode)
                writer.add_scalar('ppo/update_approx_kl', update_info["approx_kl"], episode)
                writer.add_scalar('ppo/update_clip_fraction', update_info["clip_fraction"], episode)

    if warmup_remaining > 0:
        warmup_remaining -= 1

    # TensorBoard：简化且命名更清晰
    writer.add_scalar('env/episode_total_reward', episode_reward, episode)
    writer.add_scalar('env/episode_total_cost', episode_cost, episode)
    writer.add_scalar('env/episode_T_max', T_max, episode)
    writer.add_scalar('env/episode_steps', episode_steps, episode)
    writer.add_scalar('buffer/replay_size', replay_buffer.size(), episode)
    if episode_critic_loss != 0.0:
        writer.add_scalar('maddpg/episode_critic_loss_sum', episode_critic_loss, episode)
        writer.add_scalar('maddpg/episode_actor_loss_sum', episode_actor_loss, episode)
    writer.add_scalar('training/phase_id', 0.0 if training_who == "ppo" else 1.0, episode)
    writer.add_scalar('training/num_alternations', num_alternations, episode)
    writer.flush()

    if episode % 10 == 0 or episode < 3:
        print(f"[Ep {episode}] training={training_who}, steps={episode_steps}, "
              f"reward={episode_reward:.2f}, cost={episode_cost:.4f}, "
              f"T_max={T_max:.4f}, T_spread={T_max - T_min:.4f}, "
              f"buffer={replay_buffer.size()}, alt={num_alternations}")

    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"Episode {episode}: training={training_who}, reward={episode_reward:.4f}, "
                f"cost={episode_cost:.4f}, T_max={T_max:.4f}, steps={episode_steps}\n")
        for i in range(num_UAVs):
            f.write(f"  UAV{i}: T={uav_completion_times[i]:.4f}, "
                    f"route={' -> '.join(map(str, UAV_routes[i]))}\n")

    if ppo_assign_records:
        with open(ppo_assign_log_path, 'a', encoding='utf-8') as f:
            f.write(f"Episode {episode} (training={training_who}):\n")
            for rec in ppo_assign_records:
                f.write(rec + "\n")

    # 收敛检测（基于 CV）：PPO 阶段用 T_max（与其优化目标对齐），MADDPG 阶段用路由总成本
    if training_who == "ppo":
        _cost = float(T_max)
    else:
        _cost = float(episode_cost)
    recent_costs.append(_cost)

    phase_ep_count = episode - phase_start_episode
    _sched_idx = min(num_alternations, len(_cv_schedule) - 1)
    conv_cv = _cv_schedule[_sched_idx]
    conv_window = _window_schedule[_sched_idx]

    if len(recent_costs) > conv_window:
        recent_costs.pop(0)

    converged = False
    if phase_ep_count >= min_episodes_before_switch and len(recent_costs) >= conv_window:
        mean_c = np.mean(recent_costs)
        std_c = np.std(recent_costs)
        cv = std_c / (mean_c + 1e-8)
        if cv <= conv_cv:
            converged = True
            print(f"[收敛] {training_who} 在 episode {episode} 收敛 "
                  f"(本阶段 {phase_ep_count} ep, CV={cv:.4f} <= {conv_cv})")
            with open(seed_log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{training_who} 收敛于 episode {episode} "
                        f"(CV={cv:.4f}, mean_cost={mean_c:.4f}, alternation={num_alternations})\n")

    if converged:
        if training_who == "ppo":
            freeze_params(ppo.actor, ppo.critic)
            unfreeze_params(maddpg.actor, maddpg.critic)
            maddpg.actor_optimizer = optim.Adam(maddpg.actor.parameters(), lr=5e-5)
            maddpg.critic_optimizer = optim.Adam(maddpg.critic.parameters(), lr=5e-5)
            replay_buffer = ReplayBuffer(capacity=20000)
            warmup_remaining = warmup_episodes_after_switch
            training_who = "maddpg"
            print(f"[切换] 冻结 PPO，解冻 MADDPG 开始训练（预热 {warmup_episodes_after_switch} ep）")
        else:
            freeze_params(maddpg.actor, maddpg.critic)
            unfreeze_params(ppo.actor, ppo.critic)
            ppo.optimizer_actor = optim.Adam(ppo.actor.parameters(), lr=5e-5)
            ppo.optimizer_critic = optim.Adam(ppo.critic.parameters(), lr=5e-5)
            training_who = "ppo"
            full_cycle_count += 1
            print(f"[切换] 冻结 MADDPG，解冻 PPO 开始训练（周期 {full_cycle_count}/{max_full_cycles_per_seed}）")

        num_alternations += 1
        recent_costs = []
        phase_start_episode = episode + 1
        writer.add_scalar('training/alternation_event', num_alternations, episode)

        # 不再在两个阶段各收敛一次后自动更换客户点，始终在同一批客户点上训练

    if (episode + 1) >= max_episodes:
        print(f"已达到最大训练轮数 {max_episodes}，结束训练。")

# ==========================
# 训练结束
# ==========================
train_end_time = time.time()
total_train_seconds = train_end_time - train_start_time
print(f"总训练时长: {total_train_seconds:.2f} 秒, 交替次数: {num_alternations}")

with open(log_file_path, 'a', encoding='utf-8') as f:
    f.write(f"\nTotal training time: {total_train_seconds:.2f} seconds\n")
    f.write(f"Total alternations: {num_alternations}\n")

# 保存最终模型
save_dir = result_dir
torch.save({
    'maddpg_actor': maddpg.actor.state_dict(),
    'maddpg_critic': maddpg.critic.state_dict(),
    'maddpg_target_actor': maddpg.target_actor.state_dict(),
    'maddpg_target_critic': maddpg.target_critic.state_dict(),
    'ppo_actor': ppo.actor.state_dict(),
    'ppo_critic': ppo.critic.state_dict(),
    'obs_dim': obs_dim,
    'num_customers': num_customers,
    'num_UAVs': num_UAVs,
}, os.path.join(save_dir, 'ppo_maddpg_final.pt'))
print(f"[保存] 最终模型已保存至: {os.path.join(save_dir, 'ppo_maddpg_final.pt')}")

# 事件记录
events_path = os.path.join(result_dir, 'events.txt')
with open(events_path, 'w', encoding='utf-8') as f:
    f.write("==== Broken events ====\n")
    f.write(f"Total broken events: {len(env.broken_events)}\n")
    for ev in env.broken_events:
        f.write(str(ev) + "\n")
    f.write("\n==== Dynamic customer events ====\n")
    f.write(f"Total dynamic customer events: {len(env.dynamic_customer_events)}\n")
    for ev in env.dynamic_customer_events:
        f.write(str(ev) + "\n")

writer.close()
print(f"训练完成。日志目录: {result_dir}")
