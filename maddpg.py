"""
MADDPG 独立模块：便于单独调试 MADDPG 算法。
包含 ActorNetwork、CriticNetwork、ReplayBuffer、MADDPG 类。
主训练脚本 PPO+MADDPG.py 可改为从此文件导入 MADDPG 相关类。
"""
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn
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

        # 仅用 UAV_obs，不再单独输入客户点 xyd（UAV_obs 中已含足够信息）
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.maintenance_out = nn.Linear(128, 1)
        self.maintenance_activation = nn.Sigmoid()
        self.routing_out = nn.Linear(128, num_customers + 1)

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

        return maintenance_action, routing_probs

    def get_UAV_action_mask(self, UAV_obs_tensor_i):
        env = self.env if self.env is not None else _default_env
        if env is None:
            raise RuntimeError("ActorNetwork 需要 env（或设置 maddpg._default_env）以计算 action mask")

        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)
        num_customers = self.num_customers
        is_batch = len(UAV_obs_tensor_i.shape) == 2
        if is_batch:
            num_batch, num_feature = UAV_obs_tensor_i.shape
        else:
            num_batch = 1
            UAV_obs_tensor_i = UAV_obs_tensor_i.unsqueeze(0).to(device)
            num_feature = UAV_obs_tensor_i.shape[1]

        customer_sequences = UAV_obs_tensor_i[:, -num_customers:]
        masks = torch.zeros((num_batch, num_customers + 1), dtype=torch.bool).to(device)
        masks[:, 0] = True

        energy = UAV_obs_tensor_i[:, 4].to(device)
        broken = UAV_obs_tensor_i[:, 6].to(device)
        load = UAV_obs_tensor_i[:, 3].to(device)
        destination = UAV_obs_tensor_i[:, 1].long().to(device)

        customer_cargo_demands = torch.tensor(env.customer_cargo_demands).to(device)
        distance_matrix = torch.tensor(env.distance_matrix).to(device)
        max_UAVs_load = torch.tensor(env.max_UAVs_load).to(device)
        fix_v = torch.tensor(env.fix_v).to(device)
        energy_parameter = torch.tensor(env.config['energy_parameter']).to(device)
        max_energy = torch.tensor(env.max_energy).to(device)
        min_energy = torch.tensor(env.min_energy).to(device)

        masks[broken == 1] = torch.tensor([True] + [False] * num_customers, dtype=torch.bool).to(device)
        masks[:, 1:] = (customer_sequences == 1)
        masks[destination == 0, 0] = False

        for j in range(1, num_customers + 1):
            demand_j = customer_cargo_demands[j - 1]
            arrival_time_to_j = distance_matrix[destination, j] / fix_v
            arrival_time_to_warehouse = distance_matrix[j, 0] / fix_v
            load_to_j = (100 + load + demand_j) ** (3 / 2)
            energy_to_j = energy - energy_parameter * load_to_j * arrival_time_to_j * 0.1 / max_energy
            energy_to_warehouse = energy_to_j - energy_parameter * load_to_j * arrival_time_to_warehouse * 0.1 / max_energy
            masks[:, j] &= (load + demand_j <= max_UAVs_load) & (energy_to_warehouse > min_energy)

        if is_batch:
            masks = masks
        else:
            masks = masks.squeeze(0)
        masks_tensor = masks.to(device)
        return masks_tensor


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
        UAV_obs = UAV_obs.view(256, -1).to(device)
        maintenance_action = maintenance_action.to(device)
        routing_action = routing_action.to(device)
        selected_UAV = selected_UAV.to(device)

        x = torch.cat([UAV_obs, maintenance_action, routing_action, selected_UAV], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value


# ==========================
# 3. 经验回放 ReplayBuffer
# ==========================
class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.samp_counts = {}

    def add(self, samp_number_tensor, x_tensor, y_tensor, d_tensor, UVA_obs, maintenance_action, routing_action, reward, next_UAV_obs, done, selected_UAV):
        experience = (samp_number_tensor, x_tensor, y_tensor, d_tensor, UVA_obs, maintenance_action, routing_action, reward, next_UAV_obs, done, selected_UAV)
        samp_number = experience[0]
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.samp_counts[samp_number] = self.samp_counts.get(samp_number, 0) + 1
        else:
            removable = [
                exp for exp in self.buffer
                if self.samp_counts[exp[0]] > 200
            ]
            if removable:
                removable.sort(key=lambda exp: exp[7])
                to_remove = removable[0]
                self.buffer.remove(to_remove)
                self.samp_counts[to_remove[0]] -= 1
            self.buffer.append(experience)
            self.samp_counts[samp_number] = self.samp_counts.get(samp_number, 0) + 1

    def sample(self, batch_size):
        samp_numbers = [exp[0].item() for exp in self.buffer]
        unique_numbers = list(set(samp_numbers))
        samples_per_number = {key: [] for key in unique_numbers}
        for exp in self.buffer:
            samples_per_number[exp[0].item()].append(exp)

        gmm = GaussianMixture(n_components=len(unique_numbers), random_state=42)
        samp_numbers_array = np.array(samp_numbers).reshape(-1, 1)
        gmm.fit(samp_numbers_array)

        densities = gmm.predict_proba(samp_numbers_array)
        sampling_ratios = densities.mean(axis=0)
        sampling_ratios /= sampling_ratios.sum()

        sampled_experiences = []
        group_sample_counts = [int(batch_size * ratio) for ratio in sampling_ratios]
        group_sample_counts[-1] += batch_size - sum(group_sample_counts)

        for group_idx, (number, samples) in enumerate(samples_per_number.items()):
            group_sample_count = group_sample_counts[group_idx]
            samples = [
                [torch.tensor(item, dtype=torch.float32).to(device) if not isinstance(item, torch.Tensor) else item.to(device)
                 for item in exp]
                for exp in samples
            ]
            indices = torch.randperm(len(samples))[:min(group_sample_count, len(samples))]
            sampled_experiences.extend([samples[i] for i in indices])

        if len(sampled_experiences) < batch_size:
            samples_fill = batch_size - len(sampled_experiences)
            indices_fill = torch.randperm(len(self.buffer))[:samples_fill]
            sampled_experiences.extend([self.buffer[i] for i in indices_fill])

        sampled_indices = torch.randperm(len(sampled_experiences))[:batch_size]
        sampled_experiences = [sampled_experiences[i] for i in sampled_indices]

        unpacked = list(zip(*sampled_experiences))
        samp_number = torch.stack(unpacked[0]).long().detach().to(device)
        x_tensor = torch.stack(unpacked[1]).float().detach().to(device)
        y_tensor = torch.stack(unpacked[2]).float().detach().to(device)
        d_tensor = torch.stack(unpacked[3]).float().detach().to(device)
        UAV_obs = torch.stack(unpacked[4]).float().detach().to(device)
        maintenance_action = torch.stack(unpacked[5]).float().detach().to(device)
        routing_action = torch.stack(unpacked[6]).long().detach().to(device)
        reward = torch.tensor(unpacked[7]).float().detach().to(device)
        next_UAV_obs = torch.stack(unpacked[8]).float().to(device)
        done = torch.tensor(unpacked[9]).float().detach().to(device)
        selected_UAV = torch.stack(unpacked[10]).long().detach().to(device)

        return (
            samp_number,
            x_tensor,
            y_tensor,
            d_tensor,
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

        self.actor = ActorNetwork(obs_dim, num_customers, env=self.env).to(device)
        self.critic = CriticNetwork(obs_dim, maintenance_action_dim, routing_action_dim, selected_UAV_dim, num_customers, num_UAVs).to(device)

        self.target_actor = ActorNetwork(obs_dim, num_customers, env=self.env).to(device)
        self.target_critic = CriticNetwork(obs_dim, maintenance_action_dim, routing_action_dim, selected_UAV_dim, num_customers, num_UAVs).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, UAV_obs_tensor_i, selected_UAV):
        UAV_obs_tensor_i = UAV_obs_tensor_i.to(device)

        maintenance_action, routing_probs = self.actor(UAV_obs_tensor_i, selected_UAV)
        routing_action = torch.argmax(routing_probs, dim=-1).unsqueeze(0).to(device)

        UAV = torch.tensor(self.env.UAV_obs_matrix, dtype=torch.float32).to(device)
        broken = UAV[selected_UAV][6].to(device)
        routing_action_value = routing_action.item()

        if broken == 1:
            maintenance_action = torch.tensor([0.0], dtype=torch.float32, device=device)
        elif routing_action_value != 0:
            maintenance_action = torch.tensor([1.0], dtype=torch.float32, device=device)
        elif routing_action_value == 0:
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

        samp_number_tensor = transition_dict['samp_number_tensor'].clone().detach().to(device).float()

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

        q_value = self.critic(UAV_obs, maintenance_action, routing_action, selected_UAV).to(device)

        with torch.no_grad():
            broken_check = next_UAV_obs[:, :, 6]
            has_1 = (broken_check == 1)

            if has_1.any():
                broken_indices = torch.argmax(has_1.int(), dim=1)
            else:
                broken_indices = torch.full((next_UAV_obs.size(0),), -1, dtype=torch.long, device=next_UAV_obs.device)

            decision_time_check = next_UAV_obs[:, :, 7]
            min_indices = torch.argmin(decision_time_check, dim=1)
            next_selected_UAV = torch.where(has_1.any(dim=1), broken_indices, min_indices)
            next_selected_UAV = next_selected_UAV.unsqueeze(1)
            next_selected_UAV = next_selected_UAV.to(device)

            next_UAV_obs_next_i = next_UAV_obs[torch.arange(self.batch_size), next_selected_UAV.squeeze(), :].to(device)

            next_maintenance_action, next_routing_probs = self.actor(next_UAV_obs_next_i, next_selected_UAV)
            next_routing_action = torch.argmax(next_routing_probs, dim=-1).unsqueeze(1).to(device)

            next_q_value = self.critic(next_UAV_obs, next_maintenance_action,
                                       next_routing_action, next_selected_UAV).to(device)

            rewards = rewards.view(-1, 1).to(device)
            done = done.view(-1, 1).to(device)
            target_q_value = (rewards + self.gamma * (1 - done) * next_q_value).to(device)

        critic_loss = nn.MSELoss()(q_value, target_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        maintenance_action, routing_probs = self.actor(UAV_obs[torch.arange(self.batch_size), selected_UAV.long().squeeze(), :], selected_UAV)
        routing_action = torch.argmax(routing_probs, dim=-1).unsqueeze(1).to(device)
        actor_loss = -self.critic(UAV_obs, maintenance_action, routing_action, selected_UAV).mean()

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
    保留 env 中后半列 -2 的约定（动态客户槽）。
    """
    n_u, n_c = env.num_UAVs, env.num_customers
    env.customer_state_space = np.zeros((n_u, n_c), dtype=np.int32)
    half = n_c // 2
    for i in range(n_u):
        for j in range(half):
            if j < assignment.shape[1] and assignment[i, j] == 1:
                env.customer_state_space[i, j] = 1
    env.customer_state_space[:, half:] = -2
    env._refresh_UAV_obs()


if __name__ == "__main__":
    print("[MADDPG] 创建环境 (quiet=True，不弹窗不刷屏)...")
    _default_env = Environment(num_UAVs=3, num_customers=10, quiet=True)
    env = _default_env
    print(f"[MADDPG] Using device: {device}")

    obs_dim = env.observation_space.shape[0]
    num_customers = env.num_customers
    num_UAVs = env.num_UAVs
    maintenance_action_dim = 1
    routing_action_dim = 1

    maddpg = MADDPG(obs_dim, maintenance_action_dim, routing_action_dim, num_customers, num_UAVs, env=env)
    replay_buffer = ReplayBuffer(capacity=5000)

    update_interval = 50
    step_counter = 0
    max_episodes = 10000
    samp_number = 1

    # 收敛与训练结束
    convergence_window = 10
    convergence_threshold = 0.001
    convergence_early_stop = True
    recent_costs = []

    # 客户点轮换：收敛后换新客户点再训，共进行 num_seed_rounds 轮（每轮训到收敛再换）
    num_seed_rounds = 3
    seed_round = 0

    # TensorBoard 与日志路径
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(result_dir, exist_ok=True)
    log_dir = os.path.join(result_dir, "maddpg_standalone")
    writer = SummaryWriter(log_dir=log_dir)
    assignment_log_path = os.path.join(result_dir, "maddpg_assignment.txt")
    uav_routes_log_path = os.path.join(result_dir, "maddpg_uav_routes.txt")
    broken_episodes_log_path = os.path.join(result_dir, "maddpg_broken_episodes.txt")
    if os.path.exists(assignment_log_path):
        open(assignment_log_path, "w", encoding="utf-8").close()
    with open(uav_routes_log_path, "w", encoding="utf-8") as f:
        f.write("")  # 每 episode 追加各 UAV 路径
    with open(broken_episodes_log_path, "w", encoding="utf-8") as f:
        f.write("")  # 训练结束后写入出现损坏的 episode 及详情

    train_start_time = time.time()
    print(f"[MADDPG] 开始训练，max_episodes={max_episodes}，收敛窗口={convergence_window}，阈值={convergence_threshold}，客户点轮换轮数={num_seed_rounds}")

    for episode in range(max_episodes):
        env.current_episode = episode
        env.current_step = 0
        if episode == 0 and seed_round == 0:
            print("[MADDPG] Episode 0: 调用 env.reset()...")
        UAV_obs = env.reset()
        if episode == 0 and seed_round == 0:
            print("[MADDPG] reset() 完成，运行 GA 带容量路径规划...")
        # 遗传算法：带容量约束路径规划，解码为「客户×无人机」分配矩阵，供 MADDPG 训练
        assignment, flight_cost = get_ga_assignment(env, random_state=42)
        if episode == 0:
            print("[MADDPG] GA 完成，分配矩阵形状:", assignment.shape, "每列和(应全为1):", assignment.sum(axis=0)[: num_customers // 2])
        apply_assignment_to_env(env, assignment)
        UAV_obs = [env.UAV_obs_matrix[i].copy() for i in range(num_UAVs)]
        if episode == 0:
            print("[MADDPG] 已写入 env.customer_state_space，前半列 1 的个数:", (env.customer_state_space[:, : num_customers // 2] == 1).sum())

        log_assignment(episode, assignment, flight_cost, env, assignment_log_path)
        writer.add_scalar("ga/flight_cost", flight_cost, episode)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_critic_loss = 0.0
        episode_actor_loss = 0.0
        episode_steps = 0
        UAV_routes = [[] for _ in range(num_UAVs)]  # 本 episode 每架无人机的路径（依次访问的节点，0=仓库）
        done = False

        while not done:
            selected_UAV = env.selected_UAV()
            if episode == 0 and episode_steps == 0:
                print("[MADDPG] 首次 selected_UAV() =", selected_UAV, "（<0 会直接结束本 episode）")
            if selected_UAV < 0:
                if episode < 3:
                    print(f"[MADDPG] Episode {episode} 无有效 UAV (selected_UAV=-1)，结束本 episode，步数={episode_steps}")
                break
            selected_UAV_tensor = torch.tensor(selected_UAV, dtype=torch.float32).unsqueeze(0).to(device)
            samp_number_tensor = torch.tensor(samp_number, dtype=torch.float32).unsqueeze(0).to(device)

            UAV_obs_array = np.array(UAV_obs)
            UAV_obs_tensor = torch.tensor(UAV_obs_array, dtype=torch.float32).to(device)
            UAV_obs_tensor_i = UAV_obs_tensor[selected_UAV].clone().to(device)

            maintenance_action, routing_action = maddpg.select_action(UAV_obs_tensor_i, selected_UAV)
            next_UAV_obs, rewards, done, cost = env.step(maintenance_action, routing_action, selected_UAV)

            dest = int(routing_action.item() if hasattr(routing_action, "item") else routing_action)
            UAV_routes[selected_UAV].append(dest)

            episode_reward += float(rewards)
            episode_cost += float(cost.item() if hasattr(cost, "item") else cost)

            # 新机替换：若本步决策的 UAV 故障，用新机替代，继承其未完成任务（-1 -> 1）
            if getattr(env.UAV_state["7_broken"], "shape", None) is not None:
                broken_val = env.UAV_state["7_broken"][selected_UAV, 0]
            else:
                broken_val = env.UAV_state["7_broken"][selected_UAV]
            if int(broken_val) == 1:
                env.replace_broken_uav_with_new(selected_UAV)
                next_UAV_obs = [env.UAV_obs_matrix[i].copy() for i in range(num_UAVs)]

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(device)
            next_UAV_obs_tensor = torch.tensor(np.array(next_UAV_obs), dtype=torch.float32).to(device)

            # buffer 仍按原接口存（含 xyd 占位，网络已不消费）
            x_tensor = torch.zeros(num_customers, dtype=torch.float32).to(device)
            y_tensor = torch.zeros(num_customers, dtype=torch.float32).to(device)
            d_tensor = torch.zeros(num_customers, dtype=torch.float32).to(device)
            replay_buffer.add(
                samp_number_tensor, x_tensor, y_tensor, d_tensor,
                UAV_obs_tensor, maintenance_action, routing_action,
                rewards_tensor, next_UAV_obs_tensor, done_tensor, selected_UAV_tensor,
            )

            if step_counter % update_interval == 0 and replay_buffer.size() >= maddpg.batch_size:
                batch = replay_buffer.sample(maddpg.batch_size)
                transition_dict = {
                    "samp_number_tensor": batch[0].to(device),
                    "x_tensor": batch[1].to(device),
                    "y_tensor": batch[2].to(device),
                    "d_tensor": batch[3].to(device),
                    "UAV_obs": batch[4].to(device),
                    "maintenance_action": batch[5].to(device),
                    "routing_action": batch[6].to(device),
                    "rewards": batch[7].to(device),
                    "next_UAV_obs": batch[8].to(device),
                    "done": batch[9].to(device),
                    "selected_UAV": batch[10].to(device),
                }
                c_loss, a_loss = maddpg.update(**transition_dict)
                episode_critic_loss += c_loss
                episode_actor_loss += a_loss
                writer.add_scalar("maddpg/critic_loss_step", c_loss, step_counter)
                writer.add_scalar("maddpg/actor_loss_step", a_loss, step_counter)
                if episode < 2 and episode_steps <= update_interval * 2:
                    print(f"[MADDPG] 更新 step={step_counter} critic_loss={c_loss:.4f} actor_loss={a_loss:.4f}")

            UAV_obs = next_UAV_obs
            step_counter += 1
            episode_steps += 1
            if episode == 0 and episode_steps <= 3:
                print(f"[MADDPG] step {episode_steps}: selected_UAV={selected_UAV}, reward={rewards:.3f}, cost={float(cost):.3f}, done={done}")

        # Episode 级 TensorBoard
        writer.add_scalar("episode/total_reward", episode_reward, episode)
        writer.add_scalar("episode/total_cost", episode_cost, episode)
        writer.add_scalar("episode/critic_loss_sum", episode_critic_loss, episode)
        writer.add_scalar("episode/actor_loss_sum", episode_actor_loss, episode)
        writer.add_scalar("episode/buffer_size", replay_buffer.size(), episode)
        writer.add_scalar("episode/steps", episode_steps, episode)

        if episode % 10 == 0 or episode < 3:
            print(f"[MADDPG] Episode {episode} done, steps={episode_steps}, reward={episode_reward:.2f}, cost={episode_cost:.4f}, buffer={replay_buffer.size()}")
        writer.flush()

        # 本 episode 各无人机路径规划结果写入 txt
        with open(uav_routes_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Episode {episode} 各无人机路径（节点序列，0=仓库 1~n=客户） ===\n")
            for i in range(num_UAVs):
                f.write(f"  UAV{i}: {' -> '.join(map(str, UAV_routes[i]))}\n")

        # 收敛判断
        _cost = float(episode_cost)
        recent_costs.append(_cost)
        if len(recent_costs) > convergence_window:
            recent_costs.pop(0)
        if len(recent_costs) >= convergence_window and (max(recent_costs) - min(recent_costs) <= convergence_threshold):
            print(f"第 {seed_round + 1} 轮客户点：第 {episode} 次 episode 收敛（最近 {convergence_window} 轮 cost 波动 <= {convergence_threshold}）。")
            with open(assignment_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n第 {seed_round + 1} 轮客户点收敛于 episode {episode}。\n")
            if seed_round >= num_seed_rounds - 1:
                if convergence_early_stop:
                    print(f"已完成 {num_seed_rounds} 轮客户点训练，结束训练。")
                break
            env.update_seed()
            recent_costs = []
            seed_round += 1
            print(f"[MADDPG] 第 {seed_round + 1}/{num_seed_rounds} 轮客户点，继续训练...")
        if (episode + 1) >= max_episodes:
            print(f"已达到最大训练轮数 {max_episodes}，结束训练。")

    # 训练结束：总时长、损坏 episode 记录
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
