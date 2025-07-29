import numpy as np
import os
import matplotlib.pyplot as plt
from esn_metrics import compute_memory_capacity

# 手动写入参数
alpha_list = np.logspace(np.log10(0.01), np.log10(1), 30)
collect_len = 100000
warmup_len = 500
num_nodes = 100
spectrum_radius = 1.0
omega_list = [1]

data_dir = 'reservoir_states_data'
omega = float(omega_list[0])
mc_curve_file = f"MCcurve_N{num_nodes}_sr{spectrum_radius}_om{omega:.3f}.npz"
mc_curve_path = os.path.join(data_dir, mc_curve_file)

if os.path.exists(mc_curve_path):
    print(f"Loading MC curve from {mc_curve_path}")
    data = np.load(mc_curve_path)
    alpha_vals = data['alpha_vals']
    mc_results = data['mc_results']
else:
    print(f"[Info] MC curve file not found, computing and saving to {mc_curve_path} ...")
    maxMemory = 200
    mc_results = []
    alpha_vals = []
    for alpha in alpha_list:
        alpha_str = f"{alpha:.3f}"
        files = [f for f in os.listdir(data_dir) if f"al{alpha_str}.npz" in f]
        if not files:
            print(f"[Warning] No file found for alpha={alpha_str}")
            continue
        file_path = os.path.join(data_dir, files[0])
        data = np.load(file_path)
        input_sequence = data['input_sequence']
        reservoir_state_memory = data['reservoir_state_memory']
        MC = compute_memory_capacity(reservoir_state_memory, input_sequence, maxMemory, collect_len, warmup_len)
        mc_results.append(np.sum(MC))
        alpha_vals.append(alpha)
    alpha_vals = np.array(alpha_vals)
    mc_results = np.array(mc_results)
    idx = np.argsort(alpha_vals)
    alpha_vals = alpha_vals[idx]
    mc_results = mc_results[idx]
    np.savez(mc_curve_path, alpha_vals=alpha_vals, mc_results=mc_results)
    print(f"[Info] MC curve saved to {mc_curve_path}")

plt.figure()
plt.plot(alpha_vals, mc_results, marker='o')
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel('Memory Capacity')
plt.title('MC vs. alpha (from simulated reservoir states)')
plt.grid(True)
plt.show() 