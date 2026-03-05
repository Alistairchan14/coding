from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import gym
import random
from gym import spaces
from scipy.stats import weibull_min
from scipy.integrate import quad
import math


#参数设置   （全都丢进config）
num_UAVs = 3
num_customers = 10
fix_v = 60
c_electricity = 0.6  # 配送成本参数（电价）
c_fix = 100  # 维修成本参数
a = 100000  # 失效成本参数
p = 30  # 失效成本参数
energy_parameter = 18 # 电量计算参数
max_UAVs_load = 50 # 无人机最大载重
max_energy = energy_parameter * (100 + max_UAVs_load) ** (3/2) * (20/fix_v) * 0.001 # 无人机最大energy

#搭建环境
class Environment(gym.Env): 
    def __init__(self, num_UAVs, num_customers):
        super(Environment, self).__init__()

        self.num_UAVs = num_UAVs
        self.num_customers = num_customers
        
        self.UAVs_positions = np.zeros((num_UAVs, 2))  # 无人机起始位置在原点（0，0）
        self.fix_v = fix_v
        self.max_energy = max_energy
        self.max_UAVs_load = max_UAVs_load

        self.seed_pool = [i for i in range(5200, 120000, 1)]  # 种子池 range(start, end, step)
        self.current_seed_index = 0
        self.customer_positions = self.generate_customer_positions(self.seed_pool[self.current_seed_index])
        self.customer_cargo_demands = self.generate_customer_cargo_demands(self.seed_pool[self.current_seed_index])


        # 定义常规配置参数
        self.config = self._initialize_config()

        # 计算距离矩阵
        self.distance_matrix = self._calculate_distance_matrix()

        # 打印距离矩阵
        self._print_distance_matrix()

        # 打印客户需求
        self._print_customer_cargo_demands()

        # 打印客户需求和显示客户位置图
        self._plot_customer_positions_with_demands()

        
        # 定义状态空间UAV
        self.UAV_state_space = spaces.Dict({
            "1_position": spaces.Box(low=0, high=num_customers, shape=(num_UAVs,1), dtype=np.int32),
            "2_destination": spaces.Box(low=0, high=num_customers, shape=(num_UAVs,1), dtype=np.int32),
            "3_arrival_time": spaces.Box(low=0.0, high=24.0, shape=(num_UAVs,1), dtype=np.float32),
            "4_load": spaces.Box(low=0.0, high=200.0, shape=(num_UAVs,1), dtype=np.float32),
            "5_energy": spaces.Box(low=0.0, high=100.0, shape=(num_UAVs,1), dtype=np.float32),
            "6_age": spaces.Box(low=0.0, high=10000.0, shape=(num_UAVs,1), dtype=np.float32),
            "7_broken": spaces.Box(low=0, high=1, shape=(num_UAVs,1), dtype=np.int32),
            "8_decision_time": spaces.Box(low=0.0, high=24.0, shape=(num_UAVs,1), dtype=np.float32)
        })

        # 定义状态空间Cus
        """给定版本
        self.customer_state_space = np.array([[1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 1, 1, 1]])
        """
        self.customer_state_space = spaces.Box(low=-1,high=2,shape=(num_UAVs,num_customers),dtype=np.int32)
        #self.customer_state_space = self.new_customer_state_space()

        # 定义动作空间UAV
        self.UAV_action_space = []
        for i in range(num_UAVs):
            maintenance = spaces.Box(low=0.0, high=1.0, shape=(1,1), dtype=np.float32)
            routing = spaces.Discrete(num_customers + 1)

            UAV_action = spaces.Tuple((maintenance, routing))
            self.UAV_action_space.append(UAV_action)

        # 定义动作空间CA
        self.CA_action_space = spaces.Box(low=0, high=1, shape=(num_customers,), dtype=np.int32)  # 每个客户的分配状态 (0或1）

        # 假设观察空间包含所有UAV位置和客户位置
        obs_dim = 8 + num_customers  # 8个UAV状态 + 6个客户状态
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # 打印UAV和客户状态空间
        self._print_state_spaces()

        # 设置打印选项以调整数字精度
        np.set_printoptions(precision=2, suppress=True)

    def generate_customer_positions(self,seed):
        "生成客户点坐标"
        
        np.random.seed(seed)
        customer_positions = []  # 用于存储生成的坐标

        while len(customer_positions) < self.num_customers:
            #极坐标
            r = 10 * np.sqrt(np.random.rand())  
            theta = 2 * np.pi * np.random.rand()
            x = round(r * np.cos(theta), 2)  # 保留两位小数
            y = round(r * np.sin(theta), 2)
            customer_positions.append((x, y))

        # 转换为 numpy 数组返回
        customer_positions = np.array(customer_positions)
        return customer_positions
    
    def generate_customer_cargo_demands(self,seed):
        "生成客户需求"
        
        np.random.seed(seed)
        customer_cargo_demands = np.round(np.random.uniform(0, 45, size=self.num_customers), 2) #随机生成客户需求 [10,30)
        return customer_cargo_demands
    
    def update_seed(self):
        "切换到下一个种子，更新客户点"
        self.current_seed_index = (self.current_seed_index + 1) % len(self.seed_pool)  # 循环遍历种子池
        new_seed = self.seed_pool[self.current_seed_index]
        self.customer_positions = self.generate_customer_positions(new_seed)
        self.customer_cargo_demands = self.generate_customer_cargo_demands(new_seed)
        print(f"切换到种子: {new_seed}, 新的客户信息已更新。")


    #随机生成客户点坐标+需求，return x,y,d,customer_list
    def customer_list(self):
        customer_x = self.customer_positions[:, 0].reshape(-1, 1)  # 取所有行，第0列 (x坐标)
        customer_y = self.customer_positions[:, 1].reshape(-1, 1)  # 取所有行，第1列 (y坐标)
            
        # 提取客户需求量
        customer_d = self.customer_cargo_demands.reshape(-1, 1)  # 将需求量转换为列向量
            
        # 返回x坐标、y坐标和需求量
        return customer_x, customer_y, customer_d

    def _calculate_distance_matrix(self):
        # 仓库位置
        warehouse_position = np.array([0, 0])
        # 所有点的集合（仓库 + 客户）
        all_positions = np.vstack([warehouse_position, self.customer_positions])
        # 计算距离矩阵
        distance_matrix = np.linalg.norm(all_positions[:, np.newaxis] - all_positions, axis=2)
        return distance_matrix
    
    """
    def old_customer_state_space(self):
        from CVRPGA import CVRP
        from CVRPGA import GeneticAlgorithm
        x, y, d = self.customer_list()
        # 定义仓库（0，0），需求为 0
        warehouse = {'customer_id': 0, 'coordinate': (0, 0), 'demand': 0}

        # 随机生成 num_customers 个客户，坐标 (x, y) 在 10 以内，需求在 50kg 以内
        customer_list = [warehouse]  # 初始化客户列表，包含仓库
        for i in range(num_customers):  
            customer = {
                'customer_id': i+1,
                'coordinate': (x[i][0], y[i][0]),  # 随机坐标
                'demand': d[i][0]  # 随机需求
            }
            customer_list.append(customer)

        # 打印生成的客户列表以便查看
        print("随机生成的客户列表:")
        for customer in customer_list:
            print(customer)

        # 创建CVRP问题实例
        cvrp = CVRP(customer_list)
        # 初始化一个遗传算法实例，传入CVRP问题实例
        ga = GeneticAlgorithm(cvrp)
        assignment_matrix_np = ga.run()
        return assignment_matrix_np
    
    def new_customer_state_space(self):
        assignment_matrix_np = self.old_customer_state_space()
        
        # 获取矩阵行数和列数
        hang, lie = assignment_matrix_np.shape
        
        # 第一种情况：多余的行数分配到前 num_UAVs 行，优先分配给 1 最少的行
        if hang > num_UAVs:
            # 初始化前 num_UAVs 行的矩阵
            new_matrix = np.zeros((num_UAVs, lie), dtype=int)
            # 将前 num_UAVs 行直接复制到 new_matrix
            new_matrix[:num_UAVs] = assignment_matrix_np[:num_UAVs]
            
            # 遍历多余的行
            for i in range(num_UAVs, hang):
                # 获取每一行当前的 1 的数量
                row_sums = new_matrix.sum(axis=1)
                # 找到 1 最少的行索引
                min_indices = np.where(row_sums == row_sums.min())[0]
                # 如果有多行 1 的数量相同，随机选择一行
                target_row = min_indices[0]
                # 将当前行的任务（1 值）分配到目标行
                new_matrix[target_row] |= assignment_matrix_np[i]

            return new_matrix
        
        # 第二种情况：行数与num_UAVs相同，无需修改
        elif hang == num_UAVs:

            return assignment_matrix_np
        
        # 第三种情况：增加行数，补零到num_UAVs行
        else:  # hang < num_UAVs
            padding = np.zeros((num_UAVs - hang, lie), dtype=int)
            new_matrix = np.vstack((assignment_matrix_np, padding))

            return new_matrix
        """
        

    
    def weibull_distribution(shape, scale, size=None):
        return weibull_min.rvs(shape, scale=scale, size=size)
    
    def _initialize_config(self):
        config = {
            'c_electricity': 0.6,
            'c_fix': 100,
            'a': 100000,
            'p': 30,
            'energy_parameter': 18,  # 电量计算参数
            'weibull_shape': 1.5,  # 威布尔分布的形状参数
            'weibull_scale': 100,  # 威布尔分布的尺度参数
            'weibull_function': lambda size=None: weibull_min(config['weibull_shape'], config['weibull_scale'], size),
            # 随机故障事件（动态重规划）
            'fault_prob_per_step': 0.02,       # 每步触发随机故障的概率
            'fault_min_steps_before_trigger': 2 # 至少经过多少步后才可能触发故障
        }
        return config
    
    def _print_distance_matrix(self):
        print("\nDistance Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.distance_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title("Distance Matrix between Warehouse and Customers")
        plt.xlabel("Points (0 = Warehouse)")
        plt.ylabel("Points (0 = Warehouse)")
        plt.show()

    def _print_state_spaces(self):
        print("UAV State Space:")
        for key, space in self.UAV_state_space.spaces.items():
            print(f"{key}: {space}")

        print("\nCustomer State Space:")
        print(self.customer_state_space)

    def _print_customer_cargo_demands(self):
        print("\n客户点需求:")
        for i, demand in enumerate(self.customer_cargo_demands):
            print(f"Customer {i + 1}: {demand} kg")

    def _plot_customer_positions_with_demands(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.customer_positions[:, 0], self.customer_positions[:, 1], c='blue', marker='o', label='Customers')
        plt.scatter(0, 0, c='red', marker='s', label='Warehouse')  # 仓库位置 (0,0)

        # 标注客户位置和需求
        for i, (pos, demand) in enumerate(zip(self.customer_positions, self.customer_cargo_demands)):
            plt.text(pos[0] + 0.5, pos[1] + 0.5, f"{demand} kg", fontsize=10, color='black')
            plt.text(pos[0] + 0.2, pos[1] + 0.2, f"{i + 1}", fontsize=10, color='blue')  # 标注客户编号

        plt.title("Customer Positions and Demands")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()

    def reset(self):   
        # 1.初始状态下：位置、目的地、到达时间、负载、电量、年龄、是否损坏、决策时间
        self.UAV_state = {
            "1_position": np.zeros((self.num_UAVs, 1), dtype=np.int32),  # 出发地为仓库，初始位置为0
            "2_destination": np.zeros((self.num_UAVs, 1), dtype=np.int32),  # 没有任务分配，目的地认为为0
            "3_arrival_time": np.zeros((self.num_UAVs, 1), dtype=np.float32),  # 起始时间为0
            "4_load": np.zeros((self.num_UAVs, 1), dtype=np.float32),  # 初始负载为0
            "5_energy": np.ones((self.num_UAVs, 1), dtype=np.float32) * 100.0,  # 初始电量为100%
            "6_age": np.zeros((self.num_UAVs, 1), dtype=np.float32),  # 初始役龄为0
            "7_broken": np.zeros((self.num_UAVs, 1), dtype=np.int32),  # 无人机正常工作
            "8_decision_time": np.zeros((self.num_UAVs, 1), dtype=np.float32)  # 决策时间为0
        }

        # 客户状态重置
        #self.customer_state_space = self.new_customer_state_space()
        self.customer_state_space = np.zeros((self.num_UAVs, self.num_customers))  # 0表示无人机客户暂未匹配
        self.customer_state_space[:, self.num_customers // 2:] = -2  # 前半部分是已知客户，后半部分是随机客户标记为-2

        # 决策时刻重置→编进state里面
        # self.decision_time = np.zeros((1, 1), dtype=np.float32) 

        # 2. 重置内部变量
        self.current_step = 0  # 重置当前时间步
        self.done = False  # 重置 done 标志
        self.total_reward = 0  # 重置累计奖励
    
        # 重新计算距离矩阵
        # self.distance_matrix = distance_matrix
    
        # 打印重置后的状态和距离矩阵
        print("\nReset completed.")
    
        print("\nUAV States After Reset:")
        for key, value in self.UAV_state.items():
            print(f"{key}: {value}")
    
        print("\nCustomer State Space After Reset:")
        print(self.customer_state_space)
    
        #print("\nDistance Matrix:")
        #print(self.distance_matrix)

        # 3. 生成并返回初始观测
        UAV_obs = []
        for i in range(self.num_UAVs):
            # 获取 UAV_state 中第 i 台无人机的数据
            position = self.UAV_state["1_position"][i, 0]
            destination = self.UAV_state["2_destination"][i, 0]
            arrival_time = self.UAV_state["3_arrival_time"][i, 0]
            load = self.UAV_state["4_load"][i, 0]
            energy = self.UAV_state["5_energy"][i, 0]
            age = self.UAV_state["6_age"][i, 0]
            broken = self.UAV_state["7_broken"][i, 0]
            decision_time = self.UAV_state["8_decision_time"][i, 0]
            
            # 获取 customer_state_space 中第 i 行的数据
            customer_state = self.customer_state_space[i, :]
            
            # 组合观测数据
            o_1 = np.array([
                position,
                destination,
                arrival_time,
                load,
                energy,
                age,
                broken,
                decision_time
            ])
            o_2 = np.array([
                *customer_state  # 追加 customer_state 中的元素
            ])
            o = np.concatenate([o_1, o_2])
            UAV_obs.append(o)

            UAV_obs_matrix = np.vstack(UAV_obs)
            self.UAV_obs_matrix = UAV_obs_matrix

            # 打印观测数据
            print(f"UAV {i+1} Observation: {o}")
            print(f"UAV_obs: {UAV_obs}")
            print(f"UAV_obs: {type(UAV_obs)}")
            print(f"UAV_obs_matrix: {UAV_obs_matrix}")
            print(f"UAV_obs_matrix: {type(UAV_obs_matrix)}")
            
        return UAV_obs

    def _refresh_UAV_obs(self):
        """根据当前 UAV_state 和 customer_state_space 重新生成观测并更新 UAV_obs_matrix，返回观测列表。"""
        UAV_obs = []
        for i in range(self.num_UAVs):
            position = self.UAV_state["1_position"][i, 0]
            destination = self.UAV_state["2_destination"][i, 0]
            arrival_time = self.UAV_state["3_arrival_time"][i, 0]
            load = self.UAV_state["4_load"][i, 0]
            energy = self.UAV_state["5_energy"][i, 0]
            age = self.UAV_state["6_age"][i, 0]
            broken = self.UAV_state["7_broken"][i, 0]
            decision_time = self.UAV_state["8_decision_time"][i, 0]
            customer_state = self.customer_state_space[i, :]
            o_1 = np.array([position, destination, arrival_time, load, energy, age, broken, decision_time])
            o_2 = np.array([*customer_state])
            o = np.concatenate([o_1, o_2])
            UAV_obs.append(o)
        self.UAV_obs_matrix = np.vstack(UAV_obs)
        return UAV_obs

    def trigger_random_fault(self, probability=0.0):
        """
        以一定概率触发随机故障：随机选一架当前有任务(存在1)且未故障的无人机，设为故障并不可用，
        将其任务列由 1 改为 -1，触发上层动态重分配。故障机需回仓库维修后才会恢复。
        :param probability: 本步触发故障的概率
        :return: 若触发了故障则返回更新后的 UAV_obs 列表，否则返回 None（主循环可据此更新局部 UAV_obs）
        """
        if probability <= 0 or random.random() >= probability:
            return None
        # 找当前未故障且至少有一个任务(1)的无人机
        candidates = []
        for i in range(self.num_UAVs):
            if self.UAV_state["7_broken"][i, 0] == 1:
                continue
            if np.any(self.customer_state_space[i, :] == 1):
                candidates.append(i)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        self.UAV_state["7_broken"][chosen, 0] = 1
        # 该机所有为 1 的客户改为 -1，表示需重新分配
        self.customer_state_space[chosen, :] = np.where(
            self.customer_state_space[chosen, :] == 1, -1, self.customer_state_space[chosen, :]
        )
        print(f"[故障事件] UAV {chosen + 1} 发生故障，其任务已标记为需重分配(-1)，将优先回仓库维修。")
        return self._refresh_UAV_obs()

    def selected_UAV(self):
        valid_indices = []

        # 检查是否broken=1，优先决策故障无人机
        for idx, uav in enumerate(self.UAV_obs_matrix):
            if uav[6] == 1:  # 如果第7列的数字是1
                selected_UAV = idx  # 直接返回该行的索引
                return selected_UAV  #返回索引作为选中的无人机

        # 没有特殊情况，就按照到达时间最短的原则选择
        for idx, uav in enumerate(self.UAV_obs_matrix):
            if 1 in uav[-self.num_customers:] or uav[1] != 0:  # 检查后几列是否有1（1表示还有客户点）or没回仓库的
                valid_indices.append(idx)

        if valid_indices:
            # 从有效索引中获取第8列数据
            # min函数的目的是返回最小的元素，而不是所有满足条件的元素。
            # 如果有多个元素的值相同（例如都为0 ），min函数会返回第一个出现的简单的对应元素。
            selected_UAV = min(valid_indices, key=lambda idx: self.UAV_obs_matrix[idx][7]) #取最小决策时间
            return selected_UAV
        else:
            return -1  # 如果没有有效索引，返回None或处理其他情况


    def step(self, maintenance_action, routing_action , selected_UAV):

        # 记录当前的状态，用于后续计算
        UAV = self.UAV_state

        # 定义威布尔分布的概率密度函数 (PDF)
        def weibull_pdf(x, k, lambda_):
            if x < 0:
                return 0
            return (k / lambda_) * (x / lambda_)**(k - 1) * np.exp(-(x / lambda_)**k)

        # 定义威布尔分布的可靠性函数 (Reliability function)
        def weibull_reliability(x, k, lambda_):
            if x < 0:
                return 1
            return np.exp(-(x / lambda_)**k)

        # 维修时间
        if maintenance_action.item() == 0:
            maintenance_time = 0
        else:
            maintenance_time = 2 * (1-maintenance_action.detach().item())  

        # 更新1位置、2目的地和3到达时间
        if maintenance_action.item() == 0:
            UAV["1_position"][selected_UAV] = 0
            UAV["2_destination"][selected_UAV] = 0
            UAV["3_arrival_time"][selected_UAV] = 0
        else:
            UAV["1_position"][selected_UAV] = UAV["2_destination"][selected_UAV]
            UAV["2_destination"][selected_UAV] = routing_action.item()
            UAV["3_arrival_time"][selected_UAV] = self.distance_matrix[UAV["1_position"][selected_UAV], UAV["2_destination"][selected_UAV]] / self.fix_v + maintenance_time

        load_factor = (100 + UAV["4_load"][selected_UAV]) ** (3/2)  # 电量挂钩载重的计算
        # 更新4载重
        if routing_action.item() != 0:
            UAV["4_load"][selected_UAV] = UAV["4_load"][selected_UAV] + self.customer_cargo_demands[routing_action.item() - 1]
        else:
            UAV["4_load"][selected_UAV] = 0

        # 更新5电量
        if routing_action.item() != 0:
            UAV["5_energy"][selected_UAV] = UAV["5_energy"][selected_UAV] - self.config['energy_parameter'] * load_factor * UAV["3_arrival_time"][selected_UAV] * 0.001 / self.max_energy
        else:
            UAV["5_energy"][selected_UAV] = 100

        # 更新6役龄
        if maintenance_action.item() == 1:

            k = self.config['weibull_shape']
            lambda_ = self.config['weibull_scale']
            #分子计算
            h = UAV["3_arrival_time"][selected_UAV]
            age = UAV["6_age"][selected_UAV]
            integral, _ = quad(lambda tau: tau * weibull_pdf(age + tau, k, lambda_), 0, h)

            #分母计算
            reliability_start = weibull_reliability(age, k, lambda_)
            reliability_end = weibull_reliability(age + h, k, lambda_)
            denominator = reliability_start - reliability_end

            if denominator != 0:
                age_rate = integral / denominator
                UAV["6_age"][selected_UAV] = UAV["6_age"][selected_UAV] + age_rate
            else:
                UAV["6_age"][selected_UAV] = UAV["6_age"][selected_UAV] #避免除0错误 

        elif maintenance_action.item() > 0:
            new_age = UAV["6_age"][selected_UAV] + UAV["3_arrival_time"][selected_UAV]
            UAV["6_age"][selected_UAV] = new_age * maintenance_action.detach().item()
        else:
            UAV["6_age"][selected_UAV] = 0

        # 更新7 broken 状态
        if maintenance_action.item() == 1:
            #UAV["7_broken"][selected_UAV] = np.random.choice([0, 1], p=[0.99, 0.01])
            R = weibull_reliability(UAV["6_age"][selected_UAV], k, lambda_)
            random_broken = np.random.rand()
            if random_broken > R:
                UAV["7_broken"][selected_UAV] = 1
            else:
                UAV["7_broken"][selected_UAV] = 0
        else:
            UAV["7_broken"][selected_UAV] = 0 #只要维修了就一定不会坏

        # 更新8 decision_time 状态
        if UAV["7_broken"][selected_UAV] == 0:
            UAV["8_decision_time"][selected_UAV] = UAV["8_decision_time"][selected_UAV] + UAV["3_arrival_time"][selected_UAV]
        elif UAV["7_broken"][selected_UAV] == 1:
            UAV["8_decision_time"][selected_UAV] = UAV["8_decision_time"][selected_UAV] #故障无人机不更新决策时间，表示下次决策还是它

                
        # 更新 customer_state_space （可以变成如果broken=1的那么customer特殊对待，如果是正常的话直接1变2）
        if routing_action.item() != 0:
            if UAV["7_broken"][selected_UAV] == 1:
                # 遍历 selected_UAV 对应的行，将所有值为 1 的元素修改为 -1；若没1，则不变
                self.customer_state_space[selected_UAV, :] = np.where(self.customer_state_space[selected_UAV, :] == 1, -1, self.customer_state_space[selected_UAV, :])
            else:
                self.customer_state_space[selected_UAV, routing_action.item() - 1] = 2  # 正常服务客户点，1变2表示已服务

                # 处理随机客户：查找包含 -2 的列索引，并将前两列中的 -2 全部改为 -1
                random_customer = [j for j in range(self.customer_state_space.shape[1])
                            if -2 in self.customer_state_space[:, j]]

                # 只处理前两个有 -2 的列
                for j in random_customer[:2]:
                    self.customer_state_space[:, j] = np.where(
                        self.customer_state_space[:, j] == -2, -1, self.customer_state_space[:, j]
                    )


        elif routing_action.item() == 0:
            if UAV["7_broken"][selected_UAV] == 1:
                # 遍历 selected_UAV 对应的行，将所有值为 1 的元素修改为 -1；若没1，则不变
                self.customer_state_space[selected_UAV, :] = np.where(self.customer_state_space[selected_UAV, :] == 1, -1, self.customer_state_space[selected_UAV, :])
            else:
                self.customer_state_space[selected_UAV] = self.customer_state_space[selected_UAV] #正常回仓库不改变客户完成情况，还是这台机的任务
            # （+上层：故障所有1的点变成0初始化）

        
        # 复制其他无人机的状态
        for i in range(self.num_UAVs):
            if i != selected_UAV:
                self.UAV_state["1_position"][i] = self.UAV_state["1_position"][i]
                self.UAV_state["2_destination"][i] = self.UAV_state["2_destination"][i]
                self.UAV_state["3_arrival_time"][i] = self.UAV_state["3_arrival_time"][i]
                self.UAV_state["4_load"][i] = self.UAV_state["4_load"][i]
                self.UAV_state["5_energy"][i] = self.UAV_state["5_energy"][i]
                self.UAV_state["6_age"][i] = self.UAV_state["6_age"][i]
                self.UAV_state["7_broken"][i] = self.UAV_state["7_broken"][i]
                self.UAV_state["8_decision_time"][i] = self.UAV_state["8_decision_time"][i]
        
        # 计算奖励
        #配送成本（E-E'）（1-b）c
        delivery_cost = (self.config['energy_parameter'] * load_factor * UAV["3_arrival_time"][selected_UAV] * 0.001) * (1 - UAV["7_broken"][selected_UAV]) * self.config['c_electricity']

        # 维修成本 (该成本计算—可以用线性或者二次)
        if maintenance_action.item() == 1:
            maintenance_cost = 0
        else:
            maintenance_cost = 150000 * (maintenance_action.detach().item() - 1) ** 2 + self.config['c_fix']
        

        k = self.config['weibull_shape']
        lambda_ = self.config['weibull_scale']

        # 失效成本
        failure_cost = (self.config['a'] + self.config['p'] * UAV["4_load"][selected_UAV]) * (1 - weibull_reliability(UAV["6_age"][selected_UAV], k, lambda_))

        #总成本
        UAV_total_cost = delivery_cost + failure_cost + maintenance_cost
        cost = UAV_total_cost * 0.0001 #以万为单位

        little = 1e-5
        reward = -math.log(cost + little)

        # 获取更新后的状态观测
        new_UAV_obs = []
        for i in range(self.num_UAVs):
            # 获取 UAV_state 中第 i 台无人机的数据
            new_position = self.UAV_state["1_position"][i, 0]
            new_destination = self.UAV_state["2_destination"][i, 0]
            new_arrival_time = self.UAV_state["3_arrival_time"][i, 0]
            new_load = self.UAV_state["4_load"][i, 0]
            new_energy = self.UAV_state["5_energy"][i, 0]
            new_age = self.UAV_state["6_age"][i, 0]
            new_broken = self.UAV_state["7_broken"][i, 0]
            new_decision_time = self.UAV_state["8_decision_time"][i, 0]
            
            # 获取 customer_state_space 中第 i 行的数据
            new_customer_state = self.customer_state_space[i, :]
            
            # 组合观测数据
            o_3 = np.array([
                new_position,
                new_destination,
                new_arrival_time,
                new_load,
                new_energy,
                new_age,
                new_broken,
                new_decision_time
            ])
            o_4 = np.array([
                *new_customer_state  # 追加 customer_state 中的元素
            ])
            new_o = np.concatenate([o_3, o_4])
            new_UAV_obs.append(new_o)
            # 打印观测数据
            print(f"UAV {i+1} Observation: {new_o}")
            
            
        print(type(selected_UAV))
        # 打印更新后的状态观测
        print("Updated UAV Observations:")
        for i, obs in enumerate(new_UAV_obs):
            print(f"UAV {i+1} Observation: {obs}")

        new_UAV_obs_matrix = np.vstack(new_UAV_obs)

        # 更新全局观测
        self.UAV_obs_matrix = new_UAV_obs_matrix

        # 累计奖励
        self.total_reward += reward

        # 增加步数
        self.current_step += 1

        # 检查是否达到终止条件
        #第一个条件：检查所有无人机的状态是否都不包含 1 或 -1。第二个条件：检查 "2_destination" 的所有值是否都为 0。
        #如果这两个条件都为 True，则 done 为 True，否则为 False。
        done = (all(np.all(~np.isin(self.customer_state_space[i, :], [1, -1])) for i in range(self.num_UAVs)) and np.all(UAV["2_destination"] == 0))

        return new_UAV_obs, reward, done, cost
    
    def step_CA(self, CA_action, selected_customer):

        # 记录当前的状态，用于后续计算
        UAV = self.UAV_state

        selected_customer = int(selected_customer.item())  # 将 selected_customer 转换为整数
        self.customer_state_space[:, selected_customer] = CA_action.cpu().detach().numpy()  # 更新客户状态空间

        # 获取更新后的状态观测
        new_UAV_obs = []
        for i in range(self.num_UAVs):
            # 获取 UAV_state 中第 i 台无人机的数据
            new_position = self.UAV_state["1_position"][i, 0]
            new_destination = self.UAV_state["2_destination"][i, 0]
            new_arrival_time = self.UAV_state["3_arrival_time"][i, 0]
            new_load = self.UAV_state["4_load"][i, 0]
            new_energy = self.UAV_state["5_energy"][i, 0]
            new_age = self.UAV_state["6_age"][i, 0]
            new_broken = self.UAV_state["7_broken"][i, 0]
            new_decision_time = self.UAV_state["8_decision_time"][i, 0]
            
            # 获取 customer_state_space 中第 i 行的数据
            new_customer_state = self.customer_state_space[i, :]
            
            # 组合观测数据
            o_3 = np.array([
                new_position,
                new_destination,
                new_arrival_time,
                new_load,
                new_energy,
                new_age,
                new_broken,
                new_decision_time
            ])
            o_4 = np.array([
                *new_customer_state  # 追加 customer_state 中的元素
            ])
            new_o = np.concatenate([o_3, o_4])
            new_UAV_obs.append(new_o)
            # 打印观测数据
            print(f"UAV {i+1} Observation: {new_o}")
            
            
        # 打印更新后的状态观测
        print("Updated UAV Observations:")
        for i, obs in enumerate(new_UAV_obs):
            print(f"UAV {i+1} Observation: {obs}")

        new_UAV_obs_matrix = np.vstack(new_UAV_obs)

        # 更新全局观测
        self.UAV_obs_matrix = new_UAV_obs_matrix

        # 累计奖励
        #self.total_reward += reward
        reward_CA = 0  # CA动作没有奖励
        cost_CA = 0  # CA动作没有成本

        # 增加步数
        self.current_step += 1

        # 检查是否达到终止条件
        #第一个条件：检查所有无人机的状态是否都不包含 1 或 -1。第二个条件：检查 "2_destination" 的所有值是否都为 0。
        #如果这两个条件都为 True，则 done 为 True，否则为 False。
        done = (all(np.all(~np.isin(self.customer_state_space[i, :], [1, -1])) for i in range(self.num_UAVs)) and np.all(UAV["2_destination"] == 0))

        return new_UAV_obs, reward_CA, done, cost_CA


        

    
# 创建环境
#env = Environment(num_UAVs, num_customers)












""" 
    def get_UAV_action_masks(self):
        masks = []
        
        for i in range(self.num_UAVs):
            task_sequence = self.UAV_obs_matrix[i, -6:]  # 从customer_state_space提取task_sequence
            # 初始化mask，并在开头添加一个True，表示可以回到仓库
            mask = [True] + [False] * len(task_sequence)  

            # 从UAV_obs提取相关信息
            energy = self.UAV_obs_matrix[i][4]
            broken = self.UAV_obs_matrix[i][6]
            load = self.UAV_obs_matrix[i][3]
            destination = self.UAV_obs_matrix[i][1]  # 获取目的地
            destination = int(destination)
            arrival_time = self.UAV_obs_matrix[i][2]  # 获取到达时间

            # 如果无人机损坏，屏蔽所有点
            if broken == 1:
                mask = mask
                masks.append(mask)
                continue

            if destination == 0:  # 如果上次目的地是仓库，则这次决策屏蔽仓库
                mask[0] = False
            else:
                mask[0] = True
            
            # 更新mask中与任务序列对应的部分
            for j in range(len(task_sequence)):  # 从 0 开始，到 len(task_sequence) - 1 结束迭代
                if task_sequence[j] == 1:
                    mask[j + 1] = True  # 该客户点可服务

            # routing action的一些限制条件：电量+载重
            for j in range(1, len(mask)): # 表示 j 从 1 开始，到 len(mask) - 1 结束
                if mask[j] == True:

                    # 计算载重
                    demand_j = self.customer_cargo_demands[j - 1]
                    if load + demand_j > self.max_UAVs_load: # 载重上限
                        mask[j] = False
                        continue

                    arrival_time_to_j = self.distance_matrix[destination, j] / self.fix_v  # 到达j时间
                    arrival_time_to_warehouse = self.distance_matrix[j, 0] / self.fix_v  # 从j到仓库时间
                    #计算到该点和返回仓库的总电量消耗
                    load_to_j = (100 + load + self.customer_cargo_demands[j - 1]) ** (3/2)  # 到达j的载重
                    energy_to_j = energy - self.config['energy_parameter'] * load_to_j * arrival_time_to_j * 0.001 // self.config['max_energy']  # 到达j的电量
                    energy_to_warehouse = energy_to_j - self.config['energy_parameter'] * load_to_j * arrival_time_to_warehouse * 0.001 // self.config['max_energy']  # 返回仓库的电量

                    # 如果电量不足，则屏蔽该客户点
                    if energy_to_warehouse <= 0:
                        mask[j] = False
            
            masks.append(mask)

        # 将mask转换为numpy数组以便于后续操作
        masks = np.array(masks)

        # 计算有效的routing_options
        valid_routing_options = [np.where(mask)[0] for mask in masks]
        # 将valid转换为numpy数组以便于后续操作
        valid_routing_options = np.array(valid_routing_options, dtype=object)
        
        return masks, valid_routing_options
        """