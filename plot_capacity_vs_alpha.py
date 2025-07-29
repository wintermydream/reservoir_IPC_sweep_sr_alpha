import numpy as np
import os
import matplotlib.pyplot as plt

# 参数，与数据生成一致
data_dir = 'reservoir_states_data'
num_nodes = 100
spectrum_radius_list = [0.6, 0.8, 1.0, 1.2, 1.4]
omega = 1
alpha_len = 30
alpha_list = np.logspace(np.log10(0.01), np.log10(1), alpha_len)
max_degree = 8

degree_caps_dict = {sr: [] for sr in spectrum_radius_list}
max_degree_dict = {sr: [] for sr in spectrum_radius_list}
max_delay_dict = {sr: [] for sr in spectrum_radius_list}
valid_alpha_dict = {sr: [] for sr in spectrum_radius_list}

for sr in spectrum_radius_list:
    for alpha in alpha_list:
        npz_file = f"capacity_N{num_nodes}_sr{sr:.3f}_om{omega:.3f}_al{alpha:.3f}.npz"
        npz_path = os.path.join(data_dir, npz_file)
        if not os.path.exists(npz_path):
            print(f"File not found: {npz_path}, skip.")
            continue
        data = np.load(npz_path)
        degree_capacities = data['degree_capacities']
        delay_capacities = data['delay_capacities']
        # 补齐到max_degree阶
        padded_degree = np.zeros(max_degree)
        padded_degree[:min(len(degree_capacities), max_degree)] = degree_capacities[:max_degree]
        degree_caps_dict[sr].append(padded_degree)
        # 最大 degree/delay（非零的最大 index）
        nonzero_deg = np.where(degree_capacities > 0)[0]
        max_deg = nonzero_deg[-1] + 1 if len(nonzero_deg) > 0 else 0
        nonzero_del = np.where(delay_capacities > 0)[0]
        max_del = nonzero_del[-1] + 1 if len(nonzero_del) > 0 else 0
        max_degree_dict[sr].append(max_deg)
        max_delay_dict[sr].append(max_del)
        valid_alpha_dict[sr].append(alpha)

# 画4个奇数degree的subplot（1,3,5,7）
degree_idx_list = [1, 3, 5, 7]  # Python下标，对应degree=1,3,5,7
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = plt.get_cmap('tab10').colors
for i, d in enumerate(degree_idx_list):
    ax = axes[i]
    for idx, sr in enumerate(spectrum_radius_list):
        alphas = np.array(valid_alpha_dict[sr])
        if len(alphas) == 0:
            continue
        deg_caps = np.array(degree_caps_dict[sr])
        if deg_caps.shape[0] == 0:
            continue
        y = deg_caps[:, d]
        ax.plot(alphas, y, '-o', label=f'sr={sr}', color=colors[idx % len(colors)])
    ax.set_xlabel('alpha')
    ax.set_ylabel(f'Capacity (degree={d})')
    ax.set_title(f'Degree {d}')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle('Capacity vs alpha for odd degrees (1,3,5,7) and different spectrum_radius')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 画最大 degree vs alpha
plt.figure(figsize=(8, 5))
for idx, sr in enumerate(spectrum_radius_list):
    alphas = np.array(valid_alpha_dict[sr])
    max_degs = np.array(max_degree_dict[sr])
    if len(alphas) == 0:
        continue
    plt.plot(alphas, max_degs, '-o', label=f'sr={sr}', color=colors[idx % len(colors)])
plt.xlabel('alpha')
plt.ylabel('Max Degree')
plt.title('Max Degree vs alpha (different spectrum_radius)')
plt.xscale('log')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 画最大 delay vs alpha
plt.figure(figsize=(8, 5))
for idx, sr in enumerate(spectrum_radius_list):
    alphas = np.array(valid_alpha_dict[sr])
    max_dels = np.array(max_delay_dict[sr])
    if len(alphas) == 0:
        continue
    plt.plot(alphas, max_dels, '-o', label=f'sr={sr}', color=colors[idx % len(colors)])
plt.xlabel('alpha')
plt.ylabel('Max Delay')
plt.title('Max Delay vs alpha (different spectrum_radius)')
plt.xscale('log')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show() 