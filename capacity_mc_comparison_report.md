# 对比分析：一阶 Capacity vs. Memory Capacity (MC)

## 1. 引言

本文档旨在详细对比和解析 `capacity_calculation.py` 计算的**一阶 Capacity**与 `esn_metrics.py` 计算的 **Memory Capacity (MC)**。尽管两者在理论上都旨在衡量 ESN 对过去输入的线性记忆能力，但在实际计算中，结果可能存在显著差异。本文将从数学定义、理论等价性、实现细节和数值稳定性等多个角度，深入剖析差异的根源。

## 2. 数学定义

### 2.1 Jaeger Memory Capacity (MC)

Jaeger 的 Memory Capacity (MC) 是衡量 ESN 对过去输入 \(u(t-k)\) 的线性记忆能力的经典指标。其核心思想是，对于每一个时间延迟 \(k\)，训练一个线性 readout \(y_k(t) = \mathbf{w}_k^T \mathbf{x}(t)\) 来最优地预测 \(k\) 步前的输入 \(u(t-k)\)。

- **单个延迟的 MC**：
  \(MC_k\) 定义为预测输出 \(y_k(t)\) 与真实目标 \(u(t-k)\) 之间的相关系数的平方：
  \[
  MC_k = \frac{\text{Cov}^2(y_k(t), u(t-k))}{\text{Var}(y_k(t)) \cdot \text{Var}(u(t-k))}
  \]
  这等价于线性回归的决定系数 \(R^2\)。

- **总 MC**：
  总 MC 是对所有考察的延迟 \(k\) 的 \(MC_k\) 的总和：
  \[
  \text{Total MC} = \sum_{k=1}^{K} MC_k
  \]

在 `esn_metrics.py` 中，通过求解线性方程 \(\mathbf{W} \mathbf{X} = \mathbf{U}_{\text{past}}\) 来找到最优权重 \(\mathbf{W}\)，其中 \(\mathbf{X}\) 是 reservoir states 矩阵，\(\mathbf{U}_{\text{past}}\) 是历史输入矩阵。

### 2.2 一阶 Legendre Capacity

`capacity_calculation.py` 提供了一个更广义的框架，可以用任意正交基函数来计算不同阶数的非线性记忆容量。

- **一阶 Capacity**：
  当 `degree=1` 时，使用的基函数是**一阶 Legendre 多项式** \(P_1(u) = u\)。此时，目标函数就是输入本身。
  对于延迟 \(k\)，一阶 Capacity \(C_{1,k}\) 的计算方式与 \(MC_k\) 相同：
  \[
  C_{1,k} = \frac{\text{Cov}^2(y_k(t), P_1(u(t-k)))}{\text{Var}(y_k(t)) \cdot \text{Var}(P_1(u(t-k)))} = \frac{\text{Cov}^2(y_k(t), u(t-k))}{\text{Var}(y_k(t)) \cdot \text{Var}(u(t-k))}
  \]

- **总一阶 Capacity**：
  同样地，总一阶 Capacity 是对所有延迟 \(k\) 的 \(C_{1,k}\) 的总和：
  \[
  \text{Total 1st-degree Capacity} = \sum_{k=1}^{K} C_{1,k}
  \]

## 3. 理论等价性分析

从数学定义上看，**当且仅当 `degree=1` 时，一阶 Legendre Capacity 与 Jaeger MC 的目标是完全相同的**：衡量系统对输入的线性记忆能力。因此，理论上，两者的计算结果应该是等价的。

\[
\sum_{k=1}^{K} MC_k \quad \overset{\text{理论上}}{\equiv} \quad \sum_{k=1}^{K} C_{1,k}
\]

然而，这种等价性依赖于一系列严格的假设：
1.  **输入分布**：Legendre 多项式的正交性严格依赖于输入在 `[-1, 1]` 区间上的**均匀分布**。
2.  **实现细节**：包括数据预处理（如归一化）、样本选择、数值计算方法等必须完全一致。

## 4. 实际实现差异分析：为何结果不同？

实际代码实现中的细微差别是导致最终结果不同的根本原因。

### 4.1 数据归一化 (Normalization)

这是最主要的差异来源之一。

- **`capacity_calculation.py`**:
  在 `CapacityIterator.task()` 方法中，生成的目标函数 **强制进行了归一化**：
  ```python
  output -= output.mean()
  output /= output.std()
  ```
  这意味着，无论原始输入 \(u(t-k)\) 的统计特性如何，用于计算 Capacity 的目标始终是零均值、单位方差的。

- **`esn_metrics.py`**:
  `compute_memory_capacity` 方法直接使用原始的 `input_sequence` 来构建 `pastInputTrainMtx`，**没有进行任何归一化处理**。

**影响**：相关系数的计算对数据的尺度和平移不敏感，但协方差和方差则敏感。更重要的是，回归权重的求解会受到目标数据尺度的影响，虽然最终的 \(R^2\) 理论上应不受影响，但在数值计算上，这引入了不一致性。

### 4.2 数值计算方法

- **`esn_metrics.py`**:
  直接使用 `np.linalg.pinv(reservoir_states)` 计算伪逆来求解权重。这是一种直接、稳定、广泛使用的方法。

- **`capacity_calculation.py`**:
  通过 `scipy.linalg.pinv` 计算**状态协方差矩阵的逆** `R_inv`。然后用 `(P.T @ R_inv @ P)` 的形式计算得分。虽然这在数学上等价于基于最小二乘的解，但在高维或病态（ill-conditioned）情况下，计算协方差矩阵再求逆，可能比直接对状态矩阵求伪逆引入更多的数值误差。

### 4.3 样本切片 (Sample Slicing)

- **`capacity_calculation.py`**:
  `CapacityIterator` 的样本选择非常灵活，依赖于 `delay`, `window` 等多个参数。在 `collect` 方法中，用于计算的 `estates` 切片为 `estates[self.delay + self.window - 2:, :]`。这导致有效样本的起始点和长度随着 `delay` 的变化而动态改变。

- **`esn_metrics.py`**:
  样本选择是固定的。`pastInputTrainMtx` 和 `reservoir_states` 都使用了固定的 `trainLen` 和 `initLen`。所有 `delay` 的计算都基于同一段 `trainLen` 的数据。

**影响**：不同的样本集必然导致不同的统计结果和回归权重，最终的 Capacity/MC 值也会不同。

## 5. 总结与建议

| 特性         | `esn_metrics.py` (MC)                             | `capacity_calculation.py` (1st-degree Capacity)          | 差异影响                                                     |
| :----------- | :------------------------------------------------ | :------------------------------------------------------- | :----------------------------------------------------------- |
| **目标函数** | \(u(t-k)\)                                        | \(P_1(u(t-k)) = u(t-k)\)                                 | 理论上等价                                                   |
| **归一化**   | **无**，使用原始输入                              | **强制归一化** (零均值，单位方差)                        | **主要差异来源**。改变了目标的统计属性。                     |
| **数值方法** | 直接对状态矩阵求伪逆 `pinv(X)`                    | 对状态协方差矩阵求逆 `pinv(X.T @ X)`                     | 数值稳定性可能不同，尤其是在高维或病态条件下。               |
| **样本选择** | 固定长度 `trainLen`                               | 随 `delay` 和 `window` 动态变化                          | 导致用于计算的样本集不一致，结果自然不同。                   |

### 建议

要使两者结果对齐，建议采取以下措施进行实验：
1.  **统一归一化**：在 `esn_metrics.py` 中也对 `pastInputTrainMtx` 的每一行进行标准化，或在 `capacity_calculation.py` 中移除强制归一化步骤。
2.  **统一输入分布**：确保 `generate_reservoir_states.py` 生成的输入严格服从 `[-1, 1]` 均匀分布，以满足 Legendre 多项式的正交性前提。
3.  **统一计算逻辑**：修改其中一个脚本，使其计算逻辑（特别是样本切片和数值方法）与另一个完全一致。

通过以上调整，两者计算出的线性记忆容量将趋于一致。 