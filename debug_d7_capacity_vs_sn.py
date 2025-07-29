"""Compute and plot degree-1 and degree-3 Legendre memory capacity vs spectral
radius using capacity_calculation.CapacityIterator.

If the result file already exists in `debug_capacity_data/`, computation is
skipped and the script just plots the stored data. Otherwise it iterates over
spectral radii, calculates capacities and stores the results for future use.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import capacity_calculation

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute degree-1 & degree-3 capacity vs spectral radius")
    parser.add_argument('--data_dir', default='reservoir_states_data', help='directory with res_*.npz files')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--sn_start', type=float, default=0.0)
    parser.add_argument('--sn_end', type=float, default=0.5)
    parser.add_argument('--sn_step', type=float, default=0.05)
    parser.add_argument('--omega', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--sr', type=float, default=1.4)
    parser.add_argument('--out_dir', default='debug_capacity_data')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"d1_d3_d5_d7_capacity_vs_sn_N{args.num_nodes}_sr{args.sr:.3f}_al{args.alpha:.3f}_om{args.omega:.3f}.npz"

    if out_npz.exists():
        print(f"[Info] Cached result {out_npz.name} found – loading instead of recomputing …")
        data = np.load(out_npz)
        sn_values = data['sn_values']
        d1_scores = data['d1_scores']
        d3_scores = data['d3_scores']
        d5_scores = data['d5_scores']
        d7_scores = data['d7_scores']
    else:
        sn_values = np.arange(args.sn_start, args.sn_end + 1e-8, args.sn_step)
        d1_scores: list[float] = []
        d3_scores: list[float] = []
        d5_scores: list[float] = []
        d7_scores: list[float] = []

        data_dir = Path(args.data_dir)
        for sn in sn_values:
            fname = f"res_N{args.num_nodes}_sr{args.sr:.3f}_om{args.omega:.3f}_al{args.alpha:.3f}_sn{sn:.3f}.npz"
            fpath = data_dir / fname
            if not fpath.exists():
                print(f"[WARN] missing {fname}, skipping")
                continue
            loaded = np.load(fpath)
            inputs: np.ndarray = loaded['input_sequence']
            states: np.ndarray = loaded['reservoir_state_memory']
            if states.shape[0] == args.num_nodes:
                states = states.T

            t0 = time.time()
            """Return the aggregated degree-1 capacity for the given states/inputs."""
            # Ensure shapes: states (T, N), inputs (T,)
            if states.shape[0] != inputs.shape[0]:
                min_len = min(states.shape[0], inputs.shape[0])
                states = states[-min_len:, :]
                inputs = inputs[-min_len:]

            # Configure iterator for degree 1 only
            cap_iter = capacity_calculation.CapacityIterator(mindeg=1, maxdeg=7, m_degrees=True)
            _, all_caps, _, _ = cap_iter.collect(inputs, states)

            # Sum scores where degree == 1 (should be only those)
            d1_score = sum(c['score'] for c in all_caps if c['degree'] == 1)
            d3_score = sum(c['score'] for c in all_caps if c['degree'] == 3)
            d5_score = sum(c['score'] for c in all_caps if c['degree'] == 5)
            d7_score = sum(c['score'] for c in all_caps if c['degree'] == 7)

            elapsed = time.time() - t0
            print(f"sn={sn:.3f}: d1={d1_score:.3f}, d3={d3_score:.3f}, d5={d5_score:.3f}, d7={d7_score:.3f} (elapsed {elapsed:.2f}s)")

            d1_scores.append(d1_score)
            d3_scores.append(d3_score)
            d5_scores.append(d5_score)
            d7_scores.append(d7_score)

        sn_values = sn_values[:len(d1_scores)]
        d1_scores = np.array(d1_scores)
        d3_scores = np.array(d3_scores)
        d5_scores = np.array(d5_scores)
        d7_scores = np.array(d7_scores)

        np.savez(out_npz, sn_values=sn_values, d1_scores=d1_scores, d3_scores=d3_scores, d5_scores=d5_scores,
                 d7_scores=d7_scores, alpha=args.alpha, omega=args.omega, num_nodes=args.num_nodes)
        print(f"Saved results to {out_npz}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(sn_values, d1_scores, 'o-', label='degree 1')
    plt.plot(sn_values, d3_scores, 's--', label='degree 3')
    plt.plot(sn_values, d5_scores, 'd:', label='degree 5')
    plt.plot(sn_values, d7_scores, '^-', label='degree 7')

    plt.xlabel('Noise level')
    plt.ylabel('Capacity')
    plt.title('Degree-1, Degree-3, Degree-5, and Degree-7 capacity vs noise level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
