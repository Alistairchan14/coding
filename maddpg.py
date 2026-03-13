"""
MADDPG 独立模块：便于单独调试 MADDPG 算法。
包含 ActorNetwork、CriticNetwork、ReplayBuffer、MADDPG 类。
主训练脚本 PPO+MADDPG.py 可改为从此文件导入 MADDPG 相关类。
"""
import os
import time
import json
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.utils.tensorboard import SummaryWriter

# 环境导入（单独调试或主脚本传入）
from environment_up import Environment

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认环境（仅在被直接运行或未传入 env 时使用）
_default_env = None


def get_device():
    return device


# ==========================
# 1. Actor 网络
# ==========================
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, num_customers, env=None):
        super(ActorNetwork, self).__init__()
        self.env = env
        self.num_customers = num_customers

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.maintenance_out = nn.Linear(128, 1)
        self.maintenance_activation = nn.Sigmoid()
        self.routing_out = nn.Linear(128, num_customers + 1)

        self._env_cache_version = -1
        if env is not None:
            self._cache_env_tensors()

    def _cache_env_tensors(self):
        env = self.env if self.env is not None else _default_env
        if env is None:
            return
        self._c_demands = torch.tensor(env.customer_cargo_demands, dtype=torch.float32, device=device)
        self._c_dist_matrix = torch.tensor(env.distance_matrix, dtype=torch.float32, device=device)
        self._c_max_load = torch.tensor(env.max_UAVs_load, dtype=torch.float32, device=device)
        self._c_fix_v = torch.tensor(env.fix_v, dtype=torch.float32, device=device)
        self._c_energy_param = torch.tensor(env.config['energy_parameter'], dtype=torch.float32, device=device)
        self._c_max_energy = torch.tensor(env.max_energy, dtype=torch.float32, device=device)
        self._c_min_energy = torch.tensor(env.min_energy, dtype=torch.float32, device=device)
        self._c_cust_nodes = torch.arange(1, self.num_customers + 1, device=device)
        self._c_broken_mask_row = torch.tensor([True] + [False] * self.num_customers, dtype=torch.bool, device=device)
        self._env_cache_version += 1

    def forward(self, UAV_obs_tensor_i, selected_UAV):
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)
        masks_tensor = self.get_UAV_action_mask(UAV_obs_tensor_i)
        masks_tensor = masks_tensor.to(device)

        x = UAV_obs_tensor_i
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        maintenance_action = self.maintenance_activation(self.maintenance_out(x))
        routing_logits = self.routing_out(x)
        routing_logits = routing_logits * masks_tensor + ~masks_tensor * (-1e9)
        routing_probs = torch.softmax(routing_logits, dim=-1)

        return maintenance_action, routing_probs, routing_logits

    def get_UAV_action_mask(self, UAV_obs_tensor_i):
        if not hasattr(self, '_c_demands') or self._env_cache_version < 0:
            env = self.env if self.env is not None else _default_env
            if env is None:
                raise RuntimeError("ActorNetwork 需要 env（或设置 maddpg._default_env）以计算 action mask")
            self._cache_env_tensors()

        num_customers = self.num_customers
        is_batch = UAV_obs_tensor_i.dim() == 2
        if not is_batch:
            UAV_obs_tensor_i = UAV_obs_tensor_i.unsqueeze(0)
        num_batch = UAV_obs_tensor_i.size(0)

        customer_sequences = UAV_obs_tensor_i[:, -num_customers:]
        masks = torch.zeros((num_batch, num_customers + 1), dtype=torch.bool, device=device)
        masks[:, 0] = True

        energy = UAV_obs_tensor_i[:, 4]
        broken = UAV_obs_tensor_i[:, 6]
        load = UAV_obs_tensor_i[:, 3]
        destination = UAV_obs_tensor_i[:, 1].long()

        masks[:, 1:] = (customer_sequences == 1)
        masks[destination == 0, 0] = False
        masks[broken == 1] = self._c_broken_mask_row

        # Vectorized energy/load feasibility check for all customers at once
        demands = self._c_demands
        cust_nodes = self._c_cust_nodes
        dist_to_cust = self._c_dist_matrix[destination][:, cust_nodes]
        arrival_to_j = dist_to_cust / self._c_fix_v
        arrival_to_wh = self._c_dist_matrix[cust_nodes, 0] / self._c_fix_v

        load_factor_fwd = (100 + load).unsqueeze(1) ** 1.5
        load_factor_back = (100 + load.unsqueeze(1) + demands.unsqueeze(0)) ** 1.5
        e_at_j = energy.unsqueeze(1) - self._c_energy_param * load_factor_fwd * arrival_to_j * 0.001 / self._c_max_energy
        e_at_wh = e_at_j - self._c_energy_param * load_factor_back * arrival_to_wh.unsqueeze(0) * 0.001 / self._c_max_energy

        masks[:, 1:] &= ((load.unsqueeze(1) + demands.unsqueeze(0)) <= self._c_max_load) & (e_at_wh > self._c_min_energy)

        if not is_batch:
            masks = masks.squeeze(0)
        return masks


# ==========================
# 2. Critic 网络
# ==========================
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, maintenance_action_dim, routing_action_dim, selected_UAV_dim, num_customers, num_UAVs):
        super(CriticNetwork, self).__init__()
        # 仅用 UAV_obs + 动作，不再单独输入客户点 xyd
        self.fc1 = nn.Linear(
            obs_dim * num_UAVs + maintenance_action_dim + routing_action_dim + selected_UAV_dim,
            128
        )
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, UAV_obs, maintenance_action, routing_action, selected_UAV):
        batch_size = UAV_obs.size(0)
        UAV_obs = UAV_obs.view(batch_size, -1).to(device)
        maintenance_action = maintenance_action.to(device)
        routing_action = routing_action.to(device)
        selected_UAV = selected_UAV.to(device)

        x = torch.cat([UAV_obs, maintenance_action, routing_action, selected_UAV], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value


def gumbel_softmax(logits, tau=0.5, dim=-1):
    """
    Gumbel-Softmax 采样：得到可导的软 one-hot 向量，使 Critic 的梯度能回传到 Actor 的 routing 权重。
    """
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbels) / tau
    return F.softmax(y, dim=dim)


# ==========================
# 3. 经验回放 ReplayBuffer（基于 GMM 场景描述向量的分层采样）
# ==========================
class ReplayBuffer:
    """
    论文公式 (24)：场景描述向量 s = (num, x, y, d, sel)，
    拟合 H 个高斯分量的 GMM p(s) = Σ α_h N(s|μ_h,Σ_h)，
    按各分量覆盖比例做分层采样。

    使用预分配 CPU 张量存储 + 环形缓冲区，实现 O(1) add 和批量 sample。
    """

    def __init__(self, capacity, n_gmm_components=5, gmm_refit_interval=1000):
        self.capacity = capacity
        self._size = 0
        self._pos = 0
        self._fields = None
        self._scene_np = None
        self.n_gmm_components = n_gmm_components
        self.gmm_refit_interval = gmm_refit_interval
        self._cached_labels = None
        self._cached_n_comp = 1
        self._samples_since_refit = 0
        self._last_refit_size = 0
        self._cached_comp_indices = {}
        self._distinct_nums = set()
        self._distinct_sels = set()

    @staticmethod
    def _to_scalar(x):
        return x.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else float(x)

    @staticmethod
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().ravel().astype(np.float32)
        return np.asarray(x, dtype=np.float32).ravel()

    def _build_scene(self, exp):
        """s = (num, x, y, d, sel)"""
        num = self._to_scalar(exp[0])
        x = self._to_np(exp[1])
        y = self._to_np(exp[2])
        d = self._to_np(exp[3])
        sel = self._to_scalar(exp[10])
        return np.concatenate([[num], x, y, d, [sel]]).astype(np.float32)

    def _init_storage(self, experience, scene):
        self._fields = []
        for item in experience:
            if isinstance(item, torch.Tensor):
                t = item.detach().cpu()
            else:
                t = torch.tensor(item, dtype=torch.float32)
            self._fields.append(torch.zeros((self.capacity,) + t.shape, dtype=t.dtype))
        self._scene_np = np.zeros((self.capacity, len(scene)), dtype=np.float32)

    def add(self, samp_number_tensor, x_tensor, y_tensor, d_tensor,
            UVA_obs, maintenance_action, routing_action, reward,
            next_UAV_obs, done, selected_UAV):
        experience = (samp_number_tensor, x_tensor, y_tensor, d_tensor,
                      UVA_obs, maintenance_action, routing_action, reward,
                      next_UAV_obs, done, selected_UAV)
        scene = self._build_scene(experience)

        if self._fields is None:
            self._init_storage(experience, scene)

        idx = self._pos
        for f, item in enumerate(experience):
            if isinstance(item, torch.Tensor):
                self._fields[f][idx] = item.detach().cpu()
            else:
                self._fields[f][idx] = item
        self._scene_np[idx] = scene

        self._distinct_nums.add(self._to_scalar(experience[0]))
        self._distinct_sels.add(self._to_scalar(experience[10]))

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _need_refit(self):
        size_changed = abs(self._size - self._last_refit_size) > self.gmm_refit_interval
        return (self._cached_labels is None
                or self._samples_since_refit >= self.gmm_refit_interval
                or size_changed)

    def _refit_gmm(self):
        N = self._size
        scene_array = self._scene_np[:N].astype(np.float64)

        n_approx_unique = len(self._distinct_nums) * len(self._distinct_sels)
        n_comp = max(1, min(self.n_gmm_components, n_approx_unique, N))

        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag',
                               max_iter=30, random_state=42, reg_covar=1e-3)
        try:
            gmm.fit(scene_array)
            self._cached_labels = gmm.predict(scene_array)
            self._cached_n_comp = n_comp
        except ValueError:
            self._cached_labels = np.zeros(N, dtype=int)
            self._cached_n_comp = 1

        self._cached_comp_indices = {}
        for h in range(self._cached_n_comp):
            self._cached_comp_indices[h] = []
        for i, h in enumerate(self._cached_labels):
            self._cached_comp_indices[h].append(i)

        self._samples_since_refit = 0
        self._last_refit_size = N

    def sample(self, batch_size):
        N = self._size

        if self._need_refit() or self._cached_labels is None or len(self._cached_labels) != N:
            self._refit_gmm()
        self._samples_since_refit += 1

        n_comp = self._cached_n_comp
        comp_indices = self._cached_comp_indices

        ratios = np.array([len(comp_indices.get(h, [])) for h in range(n_comp)], dtype=np.float64)
        ratios /= ratios.sum() + 1e-8

        counts = [int(batch_size * r) for r in ratios]
        counts[-1] += batch_size - sum(counts)

        sampled_idx = []
        for h in range(n_comp):
            group = comp_indices.get(h, [])
            if not group:
                continue
            c = min(counts[h], len(group))
            chosen = np.random.choice(group, size=c, replace=(c > len(group)))
            sampled_idx.extend(chosen.tolist())

        if len(sampled_idx) < batch_size:
            extra = np.random.choice(N, size=batch_size - len(sampled_idx), replace=True)
            sampled_idx.extend(extra.tolist())

        np.random.shuffle(sampled_idx)
        sampled_idx = sampled_idx[:batch_size]

        indices = torch.tensor(sampled_idx, dtype=torch.long)

        samp_number = self._fields[0][indices].long().to(device)
        x_tensor = self._fields[1][indices].float().to(device)
        y_tensor = self._fields[2][indices].float().to(device)
        d_tensor = self._fields[3][indices].float().to(device)
        UAV_obs = self._fields[4][indices].float().to(device)
        maintenance_action = self._fields[5][indices].float().to(device)
        routing_action = self._fields[6][indices].long().to(device)
        reward = self._fields[7][indices].float().to(device)
        next_UAV_obs = self._fields[8][indices].float().to(device)
        done = self._fields[9][indices].float().to(device)
        selected_UAV = self._fields[10][indices].long().to(device)

        return (samp_number, x_tensor, y_tensor, d_tensor, UAV_obs,
                maintenance_action, routing_action, reward, next_UAV_obs,
                done, selected_UAV)

    def size(self):
        return self._size


# ==========================
# 4. MADDPG 算法
# ==========================
class MADDPG:
    def __init__(self, obs_dim, maintenance_action_dim, routing_action_dim, num_customers, num_UAVs, env=None,
                 lr=5e-5, gamma=0.95, buffer_capacity=5000, batch_size=256, tau=0.005):
        global _default_env
        self.env = env if env is not None else _default_env
        if self.env is None:
            raise ValueError("MADDPG 需要传入 env，或在调试时设置 maddpg._default_env")

        self.obs_dim = obs_dim
        self.maintenance_action_dim = maintenance_action_dim
        self.routing_action_dim = routing_action_dim
        self.num_customers = num_customers
        self.num_UAVs = num_UAVs
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        selected_UAV_dim = 1
        # Critic 接收 routing 的软 one-hot（num_customers+1 维）以支持 Gumbel-Softmax 梯度回传
        routing_action_dim_for_critic = num_customers + 1

        self.actor = ActorNetwork(obs_dim, num_customers, env=self.env).to(device)
        self.critic = CriticNetwork(obs_dim, maintenance_action_dim, routing_action_dim_for_critic, selected_UAV_dim, num_customers, num_UAVs).to(device)

        self.target_actor = ActorNetwork(obs_dim, num_customers, env=self.env).to(device)
        self.target_critic = CriticNetwork(obs_dim, maintenance_action_dim, routing_action_dim_for_critic, selected_UAV_dim, num_customers, num_UAVs).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, UAV_obs_tensor_i, selected_UAV, epsilon=0.0):
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)

        maintenance_action, routing_probs, _ = self.actor(UAV_obs_tensor_i, selected_UAV)

        if epsilon > 0 and np.random.rand() < epsilon:
            mask = (routing_probs > 0).float()
            valid_count = mask.sum()
            if valid_count > 0:
                uniform = mask / valid_count
                routing_action = torch.multinomial(uniform, 1).squeeze(-1).to(device)
            else:
                routing_action = torch.argmax(routing_probs, dim=-1).to(device)
        else:
            routing_action = torch.argmax(routing_probs, dim=-1).to(device)
        routing_action = routing_action.unsqueeze(0)

        broken = float(self.env.UAV_obs_matrix[selected_UAV, 6])
        routing_action_value = routing_action.item()

        if broken == 1:
            maintenance_action = torch.tensor([0.0], dtype=torch.float32, device=device)
        elif routing_action_value != 0:
            maintenance_action = torch.tensor([1.0], dtype=torch.float32, device=device)
        elif routing_action_value == 0:
            maintenance_action = maintenance_action

        return maintenance_action, routing_action

    def refresh_env_cache(self):
        self.actor._cache_env_tensors()
        self.target_actor._cache_env_tensors()

    def update(self, samp_number_tensor, x_tensor, y_tensor, d_tensor, UAV_obs, maintenance_action, routing_action, rewards, next_UAV_obs, done, selected_UAV):
        selected_UAV_f = selected_UAV.float()
        # 将 buffer 中存储的离散 routing 索引转为 one-hot，与 Critic 输入维度一致
        routing_one_hot = F.one_hot(routing_action.squeeze(-1).long(), num_classes=self.num_customers + 1).float()

        q_value = self.critic(UAV_obs, maintenance_action, routing_one_hot, selected_UAV_f)

        with torch.no_grad():
            broken_check = next_UAV_obs[:, :, 6]
            has_1 = (broken_check == 1)

            if has_1.any():
                broken_indices = torch.argmax(has_1.int(), dim=1)
            else:
                broken_indices = torch.full((next_UAV_obs.size(0),), -1, dtype=torch.long, device=device)

            decision_time_check = next_UAV_obs[:, :, 7]
            min_indices = torch.argmin(decision_time_check, dim=1)
            next_selected_UAV = torch.where(has_1.any(dim=1), broken_indices, min_indices)
            next_selected_UAV = next_selected_UAV.view(-1, 1)
            batch_sz = next_UAV_obs.size(0)
            next_UAV_obs_next_i = next_UAV_obs[torch.arange(batch_sz, device=device), next_selected_UAV.squeeze(-1), :]

            next_maintenance_action, next_routing_probs, _ = self.target_actor(next_UAV_obs_next_i, next_selected_UAV)
            next_routing_one_hot = F.one_hot(torch.argmax(next_routing_probs, dim=-1), num_classes=self.num_customers + 1).float()

            next_q_value = self.target_critic(next_UAV_obs, next_maintenance_action,
                                              next_routing_one_hot, next_selected_UAV.float())

            target_q_value = rewards.view(-1, 1) + self.gamma * (1 - done.view(-1, 1)) * next_q_value

        critic_loss = nn.MSELoss()(q_value, target_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        batch_sz = UAV_obs.size(0)
        sel = selected_UAV.long().view(-1)
        maintenance_action_a, _, routing_logits = self.actor(UAV_obs[torch.arange(batch_sz, device=device), sel, :], selected_UAV_f)
        # 使用 Gumbel-Softmax 得到可导的软 one-hot，梯度可回传到 Actor 的 routing 权重
        soft_routing = gumbel_softmax(routing_logits, tau=0.5)
        actor_loss = -self.critic(UAV_obs, maintenance_action_a, soft_routing, selected_UAV_f).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, online_network, target_network):
        for online_params, target_params in zip(online_network.parameters(), target_network.parameters()):
            target_params.data.copy_(
                target_params.data * (1.0 - self.tau) + online_params.data * self.tau
            )


# ==========================
# 5. 带容量约束路径规划的遗传算法（解码为 MADDPG 所需分配矩阵）
# ==========================
class CVRP_GA:
    """
    在不考虑随机故障的前提下，将多无人机协作揽件视为带容量约束的路径规划问题。
    染色体：客户点的访问排列（permutation）。解码时按排列顺序、在满足载重约束下拆成多条
    子路径（可多次回仓），子路径按 UAV 数量轮询分配给各机，得到「客户×无人机」分配矩阵。
    成本 = 与飞行距离相关的总成本；适应度 = 1/成本，成本越小适应度越高。
    """

    def __init__(self, distance_matrix, customer_demands, num_UAVs, max_load,
                 pop_size=80, max_gen=150, pc=0.85, pm=0.15, random_state=None):
        self.dist = np.asarray(distance_matrix, dtype=np.float64)
        self.demands = np.asarray(customer_demands, dtype=np.float64).ravel()
        self.n_c = len(self.demands)
        self.n_u = num_UAVs
        self.max_load = float(max_load)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pc = pc
        self.pm = pm
        self.rng = np.random.default_rng(random_state)

    def _decode(self, perm):
        """
        将排列 perm (length n_c) 解码为多条子路径（满足载重），再按 UAV 数轮询得到
        分配矩阵与总飞行成本。子路径内顺序即访问顺序；单条子路径超载则回仓再开新路径。
        """
        trips = []  # 每条元素为 list of 客户索引
        load = 0.0
        cur = []
        for j in perm:
            d = self.demands[j]
            if load + d <= self.max_load:
                cur.append(j)
                load += d
            else:
                if cur:
                    trips.append(cur)
                cur = [j]
                load = d
        if cur:
            trips.append(cur)

        assignment = np.zeros((self.n_u, self.n_c), dtype=np.int32)
        for idx, trip in enumerate(trips):
            u = idx % self.n_u
            for j in trip:
                assignment[u, j] = 1

        cost = 0.0
        for trip in trips:
            if not trip:
                continue
            cost += self.dist[0, trip[0] + 1]
            for i in range(len(trip) - 1):
                cost += self.dist[trip[i] + 1, trip[i + 1] + 1]
            cost += self.dist[trip[-1] + 1, 0]
        return assignment, cost

    def _fitness(self, perm):
        _, cost = self._decode(perm)
        return 1.0 / (cost + 1e-6)

    def _roulette_select(self, pop, fitness):
        f = np.array(fitness)
        f = np.maximum(f - f.min() + 1e-6, 1e-6)
        p = f / f.sum()
        idx = self.rng.choice(len(pop), size=self.pop_size, replace=True, p=p)
        return [pop[i].copy() for i in idx]

    def _crossover_ox(self, p1, p2):
        """Order Crossover (OX)：保证子代为合法排列。"""
        a, b = self.rng.integers(0, self.n_c, size=2)
        if a > b:
            a, b = b, a
        child = [-1] * self.n_c
        child[a:b] = p1[a:b]
        used = set(p1[a:b])
        fill = [x for x in p2 if x not in used]
        pos = 0
        for i in range(self.n_c):
            if child[i] == -1:
                child[i] = fill[pos]
                pos += 1
        return np.array(child, dtype=np.int32)

    def _mutate_swap(self, perm):
        perm = perm.copy()
        i, j = self.rng.integers(0, self.n_c, size=2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    def run(self):
        """运行遗传算法，返回 (assignment_matrix, total_cost)。"""
        pop = [self.rng.permutation(self.n_c).astype(np.int32) for _ in range(self.pop_size)]
        best_perm = None
        best_cost = np.inf

        for gen in range(self.max_gen):
            fitness = [self._fitness(p) for p in pop]
            for i, p in enumerate(pop):
                _, c = self._decode(p)
                if c < best_cost:
                    best_cost = c
                    best_perm = p.copy()

            pop = self._roulette_select(pop, fitness)
            next_pop = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= self.pop_size:
                    next_pop.append(pop[i].copy())
                    continue
                if self.rng.random() < self.pc:
                    c1 = self._crossover_ox(pop[i], pop[i + 1])
                    c2 = self._crossover_ox(pop[i], pop[i + 1])
                else:
                    c1, c2 = pop[i].copy(), pop[i + 1].copy()
                next_pop.append(c1)
                next_pop.append(c2)
            pop = next_pop[: self.pop_size]
            pop = [self._mutate_swap(p) if self.rng.random() < self.pm else p for p in pop]

        if best_perm is None:
            best_perm = pop[0]
            _, best_cost = self._decode(best_perm)
        assignment, cost = self._decode(best_perm)
        return assignment, float(cost)


def get_ga_assignment(env, random_state=None):
    """调用带容量约束的路径规划 GA，返回 (assignment_matrix, flight_cost)。"""
    ga = CVRP_GA(
        env.distance_matrix,
        env.customer_cargo_demands,
        env.num_UAVs,
        env.max_UAVs_load,
        pop_size=80,
        max_gen=150,
        pc=0.85,
        pm=0.15,
        random_state=random_state,
    )
    return ga.run()


def log_assignment(episode, assignment, flight_cost, env, log_path):
    """将本 episode 的任务分配结果写入日志文件。"""
    n_u, n_c = assignment.shape
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Episode {episode} GA 路径规划解码分配 ===\n")
        f.write(f"飞行成本(总路径距离): {flight_cost:.4f}\n")
        for i in range(n_u):
            assigned = np.where(assignment[i, :] == 1)[0].tolist()
            load = sum(env.customer_cargo_demands[j] for j in assigned)
            f.write(f"  UAV{i}: 客户 {assigned}, 载重 {load:.2f} / {env.max_UAVs_load}\n")
        f.write("分配矩阵 (行=UAV, 列=客户):\n")
        f.write(np.array2string(assignment, separator=", ") + "\n")


def apply_assignment_to_env(env, assignment):
    """
    将分配矩阵写入环境的 customer_state_space。
    assignment: (num_UAVs, num_customers)，(i,j)=1 表示客户 j 分配给 UAV i。
    GA 负责全部客户分配，不保留动态槽位。
    """
    n_u, n_c = env.num_UAVs, env.num_customers
    env.customer_state_space = np.zeros((n_u, n_c), dtype=np.int32)
    for i in range(n_u):
        for j in range(n_c):
            if j < assignment.shape[1] and assignment[i, j] == 1:
                env.customer_state_space[i, j] = 1
    env._refresh_UAV_obs()


def save_maddpg_weights(maddpg_agent, save_dir, obs_dim, num_customers, num_UAVs):
    """训练结束后保存 MADDPG 网络参数，供 PPO+MADDPG 加载复用。"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "maddpg_pretrained.pt")
    torch.save({
        'actor_state_dict': maddpg_agent.actor.state_dict(),
        'critic_state_dict': maddpg_agent.critic.state_dict(),
        'target_actor_state_dict': maddpg_agent.target_actor.state_dict(),
        'target_critic_state_dict': maddpg_agent.target_critic.state_dict(),
        'obs_dim': obs_dim,
        'num_customers': num_customers,
        'num_UAVs': num_UAVs,
        'maintenance_action_dim': maddpg_agent.maintenance_action_dim,
        'routing_action_dim': maddpg_agent.routing_action_dim,
        'gamma': maddpg_agent.gamma,
        'tau': maddpg_agent.tau,
    }, save_path)
    print(f"[MADDPG] 网络参数已保存至: {os.path.abspath(save_path)}")
    return save_path


def log_ga_assignment_for_ppo(seed_round, assignment, env, log_path):
    """
    将 GA 分配结果以 PPO 可读的 JSON 格式追加写入文件。
    每行一个 JSON 对象，PPO 预训练时可逐行读取，用作监督信号。
    格式：{seed_round, num_UAVs, num_customers, customer_positions, customer_demands,
           assignment_matrix, per_customer_assigned_uav}
    其中 per_customer_assigned_uav[j] = 被分配到客户 j 的 UAV 索引（one-hot → argmax），
    这正是 PPO CA 网络的目标输出。
    """
    n_u, n_c = assignment.shape
    per_customer = {}
    for j in range(n_c):
        col = assignment[:, j]
        uav_idx = int(np.argmax(col)) if col.max() > 0 else -1
        per_customer[str(j)] = uav_idx

    record = {
        "seed_round": int(seed_round),
        "num_UAVs": int(n_u),
        "num_customers": int(n_c),
        "customer_positions": env.customer_positions.tolist(),
        "customer_demands": env.customer_cargo_demands.tolist(),
        "assignment_matrix": assignment.tolist(),
        "per_customer_assigned_uav": per_customer,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_converged_result(seed_round, episode, UAV_routes, UAV_maintenance_actions,
                         env, episode_cost, log_path):
    """
    在某轮客户点收敛后，将最终路径规划结果和维修策略写入 txt。
    包含：客户信息、每架 UAV 的完整路径节点序列、每步对应的维修动作值。
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"第 {seed_round + 1} 轮客户点 收敛路径规划结果 (Episode {episode})\n")
        f.write(f"{'='*60}\n")
        f.write(f"总成本: {episode_cost:.4f}\n\n")

        f.write("客户点信息:\n")
        for j in range(env.num_customers):
            pos = env.customer_positions[j]
            demand = env.customer_cargo_demands[j]
            f.write(f"  客户{j}: 位置=({pos[0]:.2f}, {pos[1]:.2f}), 需求={demand:.2f}\n")
        f.write("\n")

        f.write("各 UAV 路径规划与维修策略:\n")
        for i in range(env.num_UAVs):
            route = UAV_routes[i]
            maint = UAV_maintenance_actions[i]
            route_str = " -> ".join(str(n) for n in route) if route else "(无动作)"
            f.write(f"  UAV{i} 路径: 仓库(0) -> {route_str} -> 仓库(0)\n")

            if maint:
                f.write(f"  UAV{i} 维修策略 (每步 maintenance_action 值):\n")
                for step_idx, (node, m_val) in enumerate(zip(route, maint)):
                    if m_val == 1.0:
                        strategy_desc = "不维修(直接飞行)"
                    elif m_val == 0.0:
                        strategy_desc = "完全维修(役龄归零)"
                    else:
                        strategy_desc = f"部分维修(维修程度={1-m_val:.2f})"
                    f.write(f"    步骤{step_idx}: 目标节点={node}, maintenance={m_val:.4f}, {strategy_desc}\n")
            f.write("\n")


if __name__ == "__main__":
    print("[MADDPG] 创建环境 (quiet=True，不弹窗不刷屏)...")
    _default_env = Environment(num_UAVs=3, num_customers=10, quiet=True)
    _default_env.config['dynamic_customer_prob'] = 0.0
    _default_env.config['dynamic_max_new_per_step'] = 0
    env = _default_env
    print(f"[MADDPG] Using device: {device}")

    obs_dim = env.observation_space.shape[0]
    num_customers = env.num_customers
    num_UAVs = env.num_UAVs
    maintenance_action_dim = 1
    routing_action_dim = 1

    maddpg = MADDPG(obs_dim, maintenance_action_dim, routing_action_dim, num_customers, num_UAVs, env=env,
                    buffer_capacity=20000, batch_size=1024)
    replay_buffer = ReplayBuffer(capacity=20000)

    update_interval = 20
    step_counter = 0
    max_episodes = 10000

    # 客户点轮换：收敛后换新客户点再训，共进行 num_seed_rounds 轮（每轮训到收敛再换）
    num_seed_rounds = 30
    seed_round = 0
    samp_number = seed_round + 1

    # epsilon-greedy 探索
    epsilon_start = 0.3
    epsilon_end = 0.05
    epsilon_decay_episodes = 2000
    epsilon_round_start = epsilon_start
    epsilon = epsilon_start

    # 收敛与训练结束
    convergence_window = 50
    convergence_threshold_cv = 0.1
    min_episodes_before_convergence = 300
    convergence_early_stop = True
    recent_costs = []

    # TensorBoard 与日志路径
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maddpgResult")
    os.makedirs(result_dir, exist_ok=True)
    print(f"[MADDPG] 结果输出目录: {os.path.abspath(result_dir)}")
    log_dir = os.path.join(result_dir, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    assignment_log_path = os.path.join(result_dir, "maddpg_assignment.txt")
    uav_routes_log_path = os.path.join(result_dir, "maddpg_uav_routes.txt")
    broken_episodes_log_path = os.path.join(result_dir, "maddpg_broken_episodes.txt")
    ga_ppo_log_path = os.path.join(result_dir, "ga_assignment_for_ppo.jsonl")
    converged_routes_log_path = os.path.join(result_dir, "maddpg_converged_routes.txt")
    if os.path.exists(assignment_log_path):
        open(assignment_log_path, "w", encoding="utf-8").close()
    with open(uav_routes_log_path, "w", encoding="utf-8") as f:
        f.write("")
    with open(broken_episodes_log_path, "w", encoding="utf-8") as f:
        f.write("")
    with open(ga_ppo_log_path, "w", encoding="utf-8") as f:
        f.write("")
    with open(converged_routes_log_path, "w", encoding="utf-8") as f:
        f.write("MADDPG 各轮客户点收敛后最终路径规划与维修策略\n")

    train_start_time = time.time()
    print(f"[MADDPG] 开始训练，max_episodes={max_episodes}，batch_size={maddpg.batch_size}，update_interval={update_interval}，客户点轮换轮数={num_seed_rounds}")

    def _run_ga_and_apply():
        assignment, flight_cost = get_ga_assignment(env, random_state=42)
        print(f"\n[GA] 第 {seed_round + 1} 轮客户点 - 任务分配结果 (飞行成本={flight_cost:.2f}):")
        for i in range(num_UAVs):
            assigned = np.where(assignment[i, :] == 1)[0].tolist()
            load = sum(env.customer_cargo_demands[j] for j in assigned)
            print(f"  UAV{i}: 客户 {assigned}, 载重 {load:.2f} / {env.max_UAVs_load}")
        print("分配矩阵:\n", assignment)
        return assignment, flight_cost

    env.reset()
    assignment, flight_cost = _run_ga_and_apply()
    apply_assignment_to_env(env, assignment)
    log_assignment(seed_round, assignment, flight_cost, env, assignment_log_path)
    log_ga_assignment_for_ppo(seed_round, assignment, env, ga_ppo_log_path)
    writer.add_scalar("ga/flight_cost", flight_cost, seed_round)
    round_start_episode = 0
    last_converged_routes = None
    last_converged_maintenance = None

    for episode in range(max_episodes):
        env.current_episode = episode
        env.current_step = 0
        UAV_obs = env.reset()
        apply_assignment_to_env(env, assignment)
        UAV_obs = [env.UAV_obs_matrix[i].copy() for i in range(num_UAVs)]

        episode_reward = 0.0
        episode_cost = 0.0
        episode_critic_loss = 0.0
        episode_actor_loss = 0.0
        episode_steps = 0
        UAV_routes = [[] for _ in range(num_UAVs)]
        UAV_maintenance_actions = [[] for _ in range(num_UAVs)]
        done = False

        ep_x_tensor = torch.tensor(env.customer_positions[:, 0], dtype=torch.float32, device=device)
        ep_y_tensor = torch.tensor(env.customer_positions[:, 1], dtype=torch.float32, device=device)
        ep_d_tensor = torch.tensor(env.customer_cargo_demands, dtype=torch.float32, device=device)
        ep_samp_tensor = torch.tensor(samp_number, dtype=torch.float32, device=device).unsqueeze(0)

        while not done:
            selected_UAV = env.selected_UAV()
            if episode == 0 and episode_steps == 0:
                print("[MADDPG] 首次 selected_UAV() =", selected_UAV, "（<0 会直接结束本 episode）")
            if selected_UAV < 0:
                if episode < 3:
                    print(f"[MADDPG] Episode {episode} 无有效 UAV (selected_UAV=-1)，结束本 episode，步数={episode_steps}")
                break
            selected_UAV_tensor = torch.tensor(selected_UAV, dtype=torch.float32, device=device).unsqueeze(0)

            UAV_obs_tensor = torch.as_tensor(np.array(UAV_obs), dtype=torch.float32, device=device)
            UAV_obs_tensor_i = UAV_obs_tensor[selected_UAV]

            maintenance_action, routing_action = maddpg.select_action(UAV_obs_tensor_i, selected_UAV, epsilon=epsilon)
            next_UAV_obs, rewards, done, cost = env.step(maintenance_action, routing_action, selected_UAV)

            dest = int(routing_action.item() if hasattr(routing_action, "item") else routing_action)
            UAV_routes[selected_UAV].append(dest)
            maint_val = float(maintenance_action.item() if hasattr(maintenance_action, "item") else maintenance_action)
            UAV_maintenance_actions[selected_UAV].append(maint_val)

            episode_reward += float(rewards)
            episode_cost += float(cost.item() if hasattr(cost, "item") else cost)

            if getattr(env.UAV_state["7_broken"], "shape", None) is not None:
                broken_val = env.UAV_state["7_broken"][selected_UAV, 0]
            else:
                broken_val = env.UAV_state["7_broken"][selected_UAV]
            if int(broken_val) == 1:
                env.replace_broken_uav_with_new(selected_UAV)
                next_UAV_obs = [env.UAV_obs_matrix[i].copy() for i in range(num_UAVs)]

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            done_tensor = torch.tensor(done, dtype=torch.float32, device=device)
            next_UAV_obs_tensor = torch.as_tensor(np.array(next_UAV_obs), dtype=torch.float32, device=device)

            replay_buffer.add(
                ep_samp_tensor, ep_x_tensor, ep_y_tensor, ep_d_tensor,
                UAV_obs_tensor, maintenance_action, routing_action,
                rewards_tensor, next_UAV_obs_tensor, done_tensor, selected_UAV_tensor,
            )

            if step_counter % update_interval == 0 and replay_buffer.size() >= maddpg.batch_size:
                num_updates = 3
                for _ in range(num_updates):
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
                writer.add_scalar("maddpg/critic_loss_step", c_loss, step_counter)
                writer.add_scalar("maddpg/actor_loss_step", a_loss, step_counter)
                if episode < 2 and episode_steps <= update_interval * 2:
                    print(f"[MADDPG] 更新 step={step_counter} x{num_updates} critic_loss={c_loss:.4f} actor_loss={a_loss:.4f}")

            UAV_obs = next_UAV_obs
            step_counter += 1
            episode_steps += 1
            if episode == 0 and episode_steps <= 3:
                print(f"[MADDPG] step {episode_steps}: selected_UAV={selected_UAV}, reward={rewards:.3f}, cost={float(cost):.3f}, done={done}")

        # Episode 级 TensorBoard
        writer.add_scalar("episode/total_reward", episode_reward, episode)
        writer.add_scalar("episode/total_cost", episode_cost, episode)
        if episode_critic_loss != 0.0 or episode_actor_loss != 0.0:
            writer.add_scalar("episode/critic_loss_sum", episode_critic_loss, episode)
            writer.add_scalar("episode/actor_loss_sum", episode_actor_loss, episode)
        writer.add_scalar("episode/buffer_size", replay_buffer.size(), episode)
        writer.add_scalar("episode/steps", episode_steps, episode)

        # epsilon 衰减（按轮内 episode 计算，换客户点后自动重新衰减）
        round_ep = episode - round_start_episode
        epsilon = max(epsilon_end, epsilon_round_start - (epsilon_round_start - epsilon_end) * round_ep / epsilon_decay_episodes)
        writer.add_scalar("episode/epsilon", epsilon, episode)

        if episode % 10 == 0 or episode < 3:
            print(f"[MADDPG] Episode {episode} done, steps={episode_steps}, reward={episode_reward:.2f}, cost={episode_cost:.4f}, eps={epsilon:.3f}, buffer={replay_buffer.size()}")
        writer.flush()

        # 本 episode 各无人机路径规划与维修策略写入 txt
        with open(uav_routes_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Episode {episode} 各无人机路径（节点序列，0=仓库 1~n=客户） ===\n")
            for i in range(num_UAVs):
                f.write(f"  UAV{i}: {' -> '.join(map(str, UAV_routes[i]))}\n")
                if UAV_maintenance_actions[i]:
                    maint_str = ", ".join(f"{v:.2f}" for v in UAV_maintenance_actions[i])
                    f.write(f"    维修动作: [{maint_str}]\n")

        last_converged_routes = UAV_routes
        last_converged_maintenance = UAV_maintenance_actions
        last_converged_cost = episode_cost

        # 收敛判断：使用变异系数 (std/mean)，窗口 50，且需至少训练 min_episodes_before_convergence
        _cost = float(episode_cost)
        recent_costs.append(_cost)
        if len(recent_costs) > convergence_window:
            recent_costs.pop(0)

        round_episode_count = episode - round_start_episode
        converged = False
        if round_episode_count >= min_episodes_before_convergence and len(recent_costs) >= convergence_window:
            mean_cost = np.mean(recent_costs)
            std_cost = np.std(recent_costs)
            cv = std_cost / (mean_cost + 1e-8)
            if cv <= convergence_threshold_cv:
                converged = True
                print(f"第 {seed_round + 1} 轮客户点：第 {episode} 次 episode 收敛（本轮已训 {round_episode_count} ep，最近 {convergence_window} 轮 CV={cv:.4f} <= {convergence_threshold_cv}，mean_cost={mean_cost:.4f}）。")
                with open(assignment_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n第 {seed_round + 1} 轮客户点收敛于 episode {episode}（CV={cv:.4f}, mean_cost={mean_cost:.4f}）。\n")

        if converged:
            log_converged_result(seed_round, episode, last_converged_routes,
                                 last_converged_maintenance, env,
                                 last_converged_cost, converged_routes_log_path)
            if seed_round >= num_seed_rounds - 1:
                if convergence_early_stop:
                    print(f"已完成 {num_seed_rounds} 轮客户点训练，结束训练。")
                break
            env.update_seed()
            seed_round += 1
            samp_number = seed_round + 1
            env.reset()
            maddpg.refresh_env_cache()
            assignment, flight_cost = _run_ga_and_apply()
            apply_assignment_to_env(env, assignment)
            log_assignment(seed_round, assignment, flight_cost, env, assignment_log_path)
            log_ga_assignment_for_ppo(seed_round, assignment, env, ga_ppo_log_path)
            writer.add_scalar("ga/flight_cost", flight_cost, seed_round)
            recent_costs = []
            round_start_episode = episode + 1
            epsilon_round_start = epsilon_start * 0.5
            epsilon = epsilon_round_start
            print(f"[MADDPG] 第 {seed_round + 1}/{num_seed_rounds} 轮客户点，epsilon 重置为 {epsilon_round_start:.2f}，继续训练...")
        if (episode + 1) >= max_episodes:
            print(f"已达到最大训练轮数 {max_episodes}，结束训练。")

    # 训练结束：保存网络参数、总时长、损坏 episode 记录
    save_maddpg_weights(maddpg, result_dir, obs_dim, num_customers, num_UAVs)

    train_end_time = time.time()
    total_train_seconds = train_end_time - train_start_time
    print(f"总训练时长: {total_train_seconds:.2f} 秒")
    with open(assignment_log_path, "a", encoding="utf-8") as f:
        f.write(f"\nTotal training time: {total_train_seconds:.2f} seconds\n")

    with open(broken_episodes_log_path, "w", encoding="utf-8") as f:
        f.write("==== 出现损坏的 Episode 记录（来自 env.broken_events）====\n")
        f.write(f"总损坏次数: {len(env.broken_events)}\n\n")
        episodes_with_damage = sorted(set(ev["episode"] for ev in env.broken_events))
        f.write(f"涉及 episode 列表: {episodes_with_damage}\n\n")
        for ev in env.broken_events:
            f.write(f"  episode={ev['episode']} step={ev['step']} uav_id={ev['uav_id']} position={ev['position']} destination={ev['destination']} age={ev['age']} energy={ev['energy']}\n")

    writer.close()
