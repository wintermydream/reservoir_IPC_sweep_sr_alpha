import numpy as np
import os
from esn_metrics import fct_reservoir_state
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='reservoir_states_data')
    parser.add_argument('--spectrum_radius_start', type=float, default=0.4)
    parser.add_argument('--spectrum_radius_end', type=float, default=1.6)
    parser.add_argument('--spectrum_radius_step', type=float, default=0.05)
    parser.add_argument('--alpha_len', type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir
    spectrum_radius_start = args.spectrum_radius_start
    spectrum_radius_end = args.spectrum_radius_end
    spectrum_radius_step = args.spectrum_radius_step
    alpha_len = args.alpha_len

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # ESN parameters (default values)
    num_nodes = 100
    collect_len = 100000
    warmup_len = 500
    total_steps = warmup_len + collect_len

    # Activation function
    activation_function = np.tanh

    # Sweep parameters
    omega_len = 1
    #alpha_list = np.logspace(np.log10(0.01), np.log10(1), alpha_len)
    alpha_list = [0.6]
    # omega_list = np.logspace(np.log10(0.1), np.log10(5), omega_len)
    omega_list = [1]
    #spectrum_radius_list = np.arange(spectrum_radius_start, spectrum_radius_end + 1e-8, spectrum_radius_step)
    spectrum_radius_list = [0.6, 1.0, 1.4]
    sigma_n_list = np.arange(0, 0.5 + 1e-8, 0.05)

    # Weight initialization (new for each spectrum_radius)
    input_weights = 2 * np.random.rand(num_nodes, 1) - 1
    random_matrix = 2 * np.random.rand(num_nodes, num_nodes) - 1
    orth_weights, _ = np.linalg.qr(random_matrix)
    rec_weights = np.random.randn(num_nodes, num_nodes) / np.sqrt(num_nodes)
    # Input sequence (new for each run)
    input_sequence = 2 * np.random.rand(total_steps) - 1

    run_count = 0
    for spectrum_radius in spectrum_radius_list:
        recurrent_weights = spectrum_radius * rec_weights
        for alpha in alpha_list:
            for omega in omega_list:
                for sigma_n in sigma_n_list:
                    output_file = f"res_N{num_nodes}_sr{spectrum_radius:.3f}_om{omega:.3f}_al{alpha:.3f}_sn{sigma_n:.3f}.npz"
                    output_path = os.path.join(output_dir, output_file)
                    if os.path.exists(output_path):
                        print(f"[Skip] Already exists: {output_file}")
                        continue
                    t0 = time.time()
                    # Simulate the ESN reservoir and collect states
                    reservoir_state_memory = fct_reservoir_state(
                        activation_function,
                        input_weights,
                        recurrent_weights,
                        orth_weights,
                        input_sequence,
                        collect_len,
                        warmup_len,
                        omega,
                        alpha,
                        sigma_n
                    )
                    np.savez(output_path, input_sequence=input_sequence, reservoir_state_memory=reservoir_state_memory)
                    run_count += 1
                    elapsed = time.time() - t0
                    print(f"[{run_count}] Saved: {output_file} (spectrum_radius={spectrum_radius:.3f}, alpha={alpha:.3f}, omega={omega:.3f}, sigma_n={sigma_n:.3f}) | Time elapsed: {elapsed:.2f} s") 