import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = 'reservoir_states_data'
num_nodes = 100
omega = 1.0
alpha = 1  # 你要对比的 alpha
spectrum_radius_start = 0.4
spectrum_radius_end = 1.6
spectrum_radius_step = 0.05

spectrum_radius_list = np.arange(spectrum_radius_start, spectrum_radius_end + 1e-8, spectrum_radius_step)

# 读取MC曲线
mc_curve_file = f"MCcurve_N{num_nodes}_al{alpha:.3f}_om{omega:.3f}.npz"
mc_curve_path = os.path.join(data_dir, mc_curve_file)
if not os.path.exists(mc_curve_path):
    raise FileNotFoundError(f"MC curve file not found: {mc_curve_path}")
mc_data = np.load(mc_curve_path)
sr_mc = mc_data['spectrum_radius_vals']
mc_results = mc_data['mc_results']

# 读取degree=1 capacity
cap1_list = []
sr_cap = []
for sr in spectrum_radius_list:
    cap_file = f"capacity_N{num_nodes}_sr{f'{sr:.3f}'}_om{omega:.3f}_al{alpha:.3f}.npz"
    cap_path = os.path.join(data_dir, cap_file)
    if not os.path.exists(cap_path):
        print(f"Skip {cap_path} (not found)")
        continue
    data = np.load(cap_path)
    delay_capacities = data['delay_capacities']
    cap1 = delay_capacities[1] if len(delay_capacities) > 1 else np.nan
    cap1_list.append(cap1)
    sr_cap.append(sr)

sr_cap = np.array(sr_cap)
cap1_list = np.array(cap1_list)

plt.figure(figsize=(8, 5))
plt.plot(sr_mc, mc_results, label='Memory Capacity (MC)', marker='o')
plt.plot(sr_cap, cap1_list, label='Degree=1 Capacity', marker='s')
plt.xlabel('Spectrum Radius')
plt.ylabel('Capacity')
plt.title(f'Capacity vs Spectrum Radius (alpha={alpha})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 