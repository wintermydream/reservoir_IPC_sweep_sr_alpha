import numpy as np
import os
import sys
import argparse
#import capacity_calculation_V2 as capacity_calculation
import capacity_calculation
import time

def compute_and_save_capacity(npz_path, alpha, omega, num_nodes, spectrum_radius, out_path):
    data = np.load(npz_path)
    input_seq = data['input_sequence']
    states = data['reservoir_state_memory']
    # shape check: [T, N] or [N, T]
    if states.shape[0] == num_nodes:
        states = states.T  # [N, T] -> [T, N]
    # 修正 input_seq 和 states 长度不一致
    min_len = min(input_seq.shape[0], states.shape[0])
    if input_seq.shape[0] != states.shape[0]:
        input_seq = input_seq[-min_len:]
        states = states[-min_len:, :]
    # capacity calculation
    t0 = time.time()
    cap_iter = capacity_calculation.CapacityIterator()
    total_capacity, all_capacities, _, _ = cap_iter.collect(input_seq, states)
    elapsed = time.time() - t0
    # degree_capacities
    if len(all_capacities) > 0:
        max_degree = max([x['degree'] for x in all_capacities])
        degree_capacities = np.zeros(max_degree + 1)
        for cap in all_capacities:
            degree_capacities[cap['degree']] += cap['score']
        max_delay = max([x['delay'] for x in all_capacities])
        delay_capacities = np.zeros(max_delay + 1)
        for cap in all_capacities:
            delay_capacities[cap['delay']] += cap['score']
    else:
        degree_capacities = np.array([])
        delay_capacities = np.array([])
    # 保存
    np.savez(out_path,
             total_capacity=total_capacity,
             degree_capacities=degree_capacities,
             delay_capacities=delay_capacities,
             alpha=alpha,
             omega=omega,
             num_nodes=num_nodes,
             spectrum_radius=spectrum_radius)
    print(f"Saved: {os.path.basename(out_path)} | Time elapsed: {elapsed:.2f} s")


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='reservoir_states_data', help='数据文件目录')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--spectrum_radius_start', type=float, default=0.4)
    parser.add_argument('--spectrum_radius_end', type=float, default=1.6)
    parser.add_argument('--spectrum_radius_step', type=float, default=0.05)
    parser.add_argument('--omega', type=float, default=1)
    args = parser.parse_args()

    data_dir = args.data_dir
    num_nodes = args.num_nodes
    omega = args.omega
    # alpha_len = 30
    # alpha_list = np.logspace(np.log10(0.01), np.log10(1), args.alpha_len)
    alpha_list = [1]

    # spectrum_radius = 1.2
    spectrum_radius_list = np.arange(args.spectrum_radius_start, args.spectrum_radius_end + 1e-8, args.spectrum_radius_step)
    for spectrum_radius in spectrum_radius_list:
        for alpha in alpha_list:
            npz_file = f"res_N{args.num_nodes}_sr{float(spectrum_radius):.3f}_om{args.omega:.3f}_al{alpha:.3f}.npz"
            npz_path = os.path.join(args.data_dir, npz_file)
            out_file = f"capacity_N{args.num_nodes}_sr{float(spectrum_radius):.3f}_om{args.omega:.3f}_al{alpha:.3f}.npz"
            out_path = os.path.join(args.data_dir, out_file)
            if os.path.exists(out_path):
                print(f"Already exists: {out_file}, skip.")
                continue
            if not os.path.exists(npz_path):
                print(f"File not found: {npz_path}")
                continue
            compute_and_save_capacity(npz_path, alpha, args.omega, args.num_nodes, spectrum_radius, out_path)