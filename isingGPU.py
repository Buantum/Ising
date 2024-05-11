import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states
import math

# 初始化参数
L =10  # 格子尺寸
J = 1  # 相互作用强度
h = 0.0  # 外部磁场
N1 = 400000  # 每个温度的重复次数
N2 = 1   # 温度点数
temperature_values = np.linspace(2.5, 5, N2)

# 每个线程处理一个任务
threads_per_block = 256
blocks_per_grid = (N1 * N2 + (threads_per_block - 1)) // threads_per_block

# 计算能量变化
@cuda.jit(device=True)
def delta_E(spins, i, j, L, J, h):
    neighbor_sum = (
        spins[(i + 1) % L, j] + spins[(i - 1) % L, j] +
        spins[i, (j + 1) % L] + spins[i, (j - 1) % L]
    )
    dE = 2 * J * spins[i, j] * neighbor_sum + 2 * h * spins[i, j]
    return dE

# Metropolis更新步骤
@cuda.jit
def metropolis_kernel(spins, temperatures, magnetizations, L, J, h, rng_states):
    idx = cuda.grid(1)
    if idx < N1 * N2:
        temp_idx = idx // N1
        run_idx = idx % N1
        temperature = temperatures[temp_idx]
        for _ in range(L**6):
            x = int(xoroshiro128p_uniform_float32(rng_states, idx) * L)
            y = int(xoroshiro128p_uniform_float32(rng_states, idx) * L)
            dE = delta_E(spins[temp_idx, run_idx], x, y, L, J, h)
            if dE < 0 or xoroshiro128p_uniform_float32(rng_states, idx) < math.exp(-dE / temperature):
                spins[temp_idx, run_idx, x, y] *= -1

        # 计算总磁化率
        magnetization = 0
        for x in range(L):
            for y in range(L):
                magnetization += spins[temp_idx, run_idx, x, y]
        magnetizations[idx] = magnetization

# 主函数
def run_simulation(L, J, h, N1, temperature_values):
    spins = np.random.choice([-1, 1], size=(N2, N1, L, L)).astype(np.int8)
    d_spins = cuda.to_device(spins)
    d_temperatures = cuda.to_device(temperature_values)
    magnetizations = np.zeros(N1 * N2, dtype=np.float32)
    d_magnetizations = cuda.to_device(magnetizations)

    rng_states = create_xoroshiro128p_states(N1 * N2, seed=1)

    metropolis_kernel[blocks_per_grid, threads_per_block](d_spins, d_temperatures, d_magnetizations, L, J, h, rng_states)

    return d_magnetizations.copy_to_host()

# 运行模拟
magnetization_values = run_simulation(L, J, h, N1, temperature_values)

# 生成温度的重复序列
temperature_repeated = np.repeat(temperature_values, N1)

# 根据磁化率的正负分类
positive_magnetizations = magnetization_values[magnetization_values > 0]
negative_magnetizations = magnetization_values[magnetization_values < 0]

# 筛选出正负磁化率对应的温度值
temperature_pos_repeated = temperature_repeated[magnetization_values > 0]
temperature_neg_repeated = temperature_repeated[magnetization_values < 0]

# 计算每个温度的平均值和标准误差
avg_pos_magnetizations = [np.mean(positive_magnetizations[temperature_pos_repeated == T]) if np.any(temperature_pos_repeated == T) else 0 for T in temperature_values]
std_pos_magnetizations = [np.std(positive_magnetizations[temperature_pos_repeated == T]) if np.any(temperature_pos_repeated == T) else 0 for T in temperature_values]

avg_neg_magnetizations = [np.mean(negative_magnetizations[temperature_neg_repeated == T]) if np.any(temperature_neg_repeated == T) else 0 for T in temperature_values]
std_neg_magnetizations = [np.std(negative_magnetizations[temperature_neg_repeated == T]) if np.any(temperature_neg_repeated == T) else 0 for T in temperature_values]

# 绘制结果
plt.figure(figsize=(10, 6))
plt.errorbar(temperature_values, avg_pos_magnetizations, yerr=std_pos_magnetizations, fmt='o', label='Positive Magnetization', capsize=5)
plt.errorbar(temperature_values, avg_neg_magnetizations, yerr=std_neg_magnetizations, fmt='o', label='Negative Magnetization', capsize=5)
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.title('Magnetization vs. Temperature')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('magnetization_vs_temperature.png', dpi=300)