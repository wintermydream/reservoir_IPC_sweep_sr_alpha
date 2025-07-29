"""Debug script to compute and plot first-degree (d=1) memory capacity
versus spectral radius using the original `capacity_calculation.py`.

This is a trimmed-down version of `compute_capacity_vs_specRadius.py` that:
1. Forces the maximum Legendre degree to 1.
2. Iterates over a list of spectral radii and loads pre-computed reservoir
   state files (`res_N{num_nodes}_sr{sr}_om{omega}_al{alpha}.npz`).
3. Computes the degree-1 capacity via `capacity_calculation.CapacityIterator`.
4. Saves the numeric results to a *temporary* directory
   (`debug_capacity_data`) and plots the curve for visual inspection.

Usage (run from project root)::

    python debug_d1_capacity_vs_sr.py --data_dir reservoir_states_data
                                      --num_nodes 100 \
                                      --spectrum_radius_start 0.4 \
                                      --spectrum_radius_end 1.6 \
                                      --spectrum_radius_step 0.05 \
                                      --omega 1 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import capacity_calculation


def compute_d1_capacity(states: np.ndarray, inputs: np.ndarray) -> float:
    """Return the aggregated degree-1 capacity for the given states/inputs."""
    # Ensure shapes: states (T, N), inputs (T,)
    if states.shape[0] != inputs.shape[0]:
        min_len = min(states.shape[0], inputs.shape[0])
        states = states[-min_len:, :]
        inputs = inputs[-min_len:]

    # Configure iterator for degree 1 only
    cap_iter = capacity_calculation.CapacityIterator(mindeg=1, maxdeg=1, m_degrees=True)
    _, all_caps, _, _ = cap_iter.collect(inputs, states)

    # Sum scores where degree == 1 (should be only those)
    d1_score = sum(c['score'] for c in all_caps if c['degree'] == 1)
    return float(d1_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute d=1 capacity vs spectral radius")
    parser.add_argument('--data_dir', type=str, default='reservoir_states_data', help='Folder with reservoir state npz')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--spectrum_radius_start', type=float, default=0.4)
    parser.add_argument('--spectrum_radius_end', type=float, default=1.6)
    parser.add_argument('--spectrum_radius_step', type=float, default=0.05)
    parser.add_argument('--omega', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_dir', type=str, default='debug_capacity_data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr_values = np.arange(args.spectrum_radius_start, args.spectrum_radius_end + 1e-8, args.spectrum_radius_step)
    d1_scores = []

    for sr in sr_values:
        npz_name = f"res_N{args.num_nodes}_sr{sr:.3f}_om{args.omega:.3f}_al{args.alpha:.3f}.npz"
        npz_path = data_dir / npz_name
        if not npz_path.exists():
            print(f"[WARN] Missing state file {npz_name}, skipping …")
            continue

        data = np.load(npz_path)
        input_seq: np.ndarray = data['input_sequence']
        states: np.ndarray = data['reservoir_state_memory']
        if states.shape[0] == args.num_nodes:
            states = states.T  # make shape (T, N)

        t0 = time.time()
        score = compute_d1_capacity(states, input_seq)
        print(f"sr={sr:.3f} -> d1 capacity {score:.3f}  (elapsed {time.time()-t0:.2f}s)")
        d1_scores.append(score)

    sr_values = np.array(sr_values[:len(d1_scores)])
    d1_scores = np.array(d1_scores)

    # Save results
    out_file = out_dir / f"d1_capacity_vs_sr_N{args.num_nodes}_al{args.alpha:.3f}_om{args.omega:.3f}.npz"
    np.savez(out_file, sr_values=sr_values, d1_scores=d1_scores, alpha=args.alpha, omega=args.omega, num_nodes=args.num_nodes)
    print(f"Saved results to {out_file}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(sr_values, d1_scores, 'o-', label='degree-1 capacity (cov_method)')
    plt.xlabel('Spectral radius')
    plt.ylabel('Capacity (degree 1)')
    plt.title('Degree-1 capacity vs spectral radius')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
