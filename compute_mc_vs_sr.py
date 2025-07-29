import numpy as np
import os
#from esn_metrics import compute_memory_capacity
from esn_metrics import fct_d1_capacity
import matplotlib.pyplot as plt

data_dir = 'reservoir_states_data'
num_nodes = 100
omega = 1.0
alpha = 1.0
collect_len = 100000
warmup_len = 500
maxMemory = 200

spectrum_radius_start = 0.4
spectrum_radius_end = 1.6
spectrum_radius_step = 0.05
spectrum_radius_list = np.arange(spectrum_radius_start, spectrum_radius_end + 1e-8, spectrum_radius_step)

#mc_curve_file = f"MCcurve_N{num_nodes}_al{alpha:.3f}_om{omega:.3f}.npz"
mc_curve_file = f"testMCcurve_N{num_nodes}_al{alpha:.3f}_om{omega:.3f}.npz"
mc_curve_path = os.path.join(data_dir, mc_curve_file)

if os.path.exists(mc_curve_path):
    print(f"Loading MC curve from {mc_curve_path}")
    data = np.load(mc_curve_path)
    sr_vals = data['spectrum_radius_vals']
    mc_results = data['mc_results']
else:
    print(f"[Info] MC curve file not found, computing and saving to {mc_curve_path} ...")
    mc_results = []
    sr_vals = []
    for sr in spectrum_radius_list:
        sr_str = f"{sr:.3f}"
        res_file = f"res_N{num_nodes}_sr{sr_str}_om{omega:.3f}_al{alpha:.3f}.npz"
        res_path = os.path.join(data_dir, res_file)
        if not os.path.exists(res_path):
            print(f"[Warning] No file found for spectrum_radius={sr_str}")
            continue
        data = np.load(res_path)
        input_sequence = data['input_sequence']
        reservoir_state_memory = data['reservoir_state_memory']
        #MC = compute_memory_capacity(reservoir_state_memory, input_sequence, maxMemory, collect_len, warmup_len)
        MC = fct_d1_capacity(reservoir_state_memory, input_sequence, maxMemory, collect_len, warmup_len)
        mc_results.append(np.sum(MC))
        sr_vals.append(sr)
    sr_vals = np.array(sr_vals)
    mc_results = np.array(mc_results)
    idx = np.argsort(sr_vals)
    sr_vals = sr_vals[idx]
    mc_results = mc_results[idx]
    np.savez(mc_curve_path, spectrum_radius_vals=sr_vals, mc_results=mc_results)
    print(f"[Info] MC curve saved to {mc_curve_path}")

# 画图
plt.figure()
plt.plot(sr_vals, mc_results, marker='o')
plt.xlabel('Spectrum Radius')
plt.ylabel('Memory Capacity')
plt.title(f'MC vs. Spectrum Radius (alpha={alpha})')
plt.grid(True)
plt.tight_layout()
plt.show() 