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


def compute_deg_capacity(states: np.ndarray, inputs: np.ndarray, degree: int) -> float:
    """Return aggregated capacity for a given Legendre *degree*."""
    if states.shape[0] != inputs.shape[0]:
        m = min(states.shape[0], inputs.shape[0])
        states = states[-m:, :]
        inputs = inputs[-m:]

    cap_iter = capacity_calculation.CapacityIterator(mindeg=degree, maxdeg=degree, m_degrees=True)
    _, caps, _, _ = cap_iter.collect(inputs, states)
    return float(sum(c['score'] for c in caps if c['degree'] == degree))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute degree-1 & degree-3 capacity vs spectral radius")
    parser.add_argument('--data_dir', default='reservoir_states_data', help='directory with res_*.npz files')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--spectrum_radius_start', type=float, default=0.4)
    parser.add_argument('--spectrum_radius_end', type=float, default=1.6)
    parser.add_argument('--spectrum_radius_step', type=float, default=0.05)
    parser.add_argument('--omega', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_dir', default='debug_capacity_data')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"d1_d3_capacity_vs_sr_N{args.num_nodes}_al{args.alpha:.3f}_om{args.omega:.3f}.npz"

    if out_npz.exists():
        print(f"[Info] Cached result {out_npz.name} found – loading instead of recomputing …")
        data = np.load(out_npz)
        sr_values = data['sr_values']
        d1_scores = data['d1_scores']
        d3_scores = data['d3_scores']
    else:
        sr_values = np.arange(args.spectrum_radius_start, args.spectrum_radius_end + 1e-8, args.spectrum_radius_step)
        d1_scores: list[float] = []
        d3_scores: list[float] = []

        data_dir = Path(args.data_dir)
        for sr in sr_values:
            fname = f"res_N{args.num_nodes}_sr{sr:.3f}_om{args.omega:.3f}_al{args.alpha:.3f}.npz"
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
            cap_iter = capacity_calculation.CapacityIterator(mindeg=1, maxdeg=3, m_degrees=True)
            _, all_caps, _, _ = cap_iter.collect(inputs, states)

            # Sum scores where degree == 1 (should be only those)
            d1_score = sum(c['score'] for c in all_caps if c['degree'] == 1)
            d3_score = sum(c['score'] for c in all_caps if c['degree'] == 3)
            elapsed = time.time() - t0
            print(f"sr={sr:.3f}: d1={d1_score:.3f}, d3={d3_score:.3f} (elapsed {elapsed:.2f}s)")

            d1_scores.append(d1_score)
            d3_scores.append(d3_score)

        sr_values = sr_values[:len(d1_scores)]
        d1_scores = np.array(d1_scores)
        d3_scores = np.array(d3_scores)
        np.savez(out_npz, sr_values=sr_values, d1_scores=d1_scores, d3_scores=d3_scores,
                 alpha=args.alpha, omega=args.omega, num_nodes=args.num_nodes)
        print(f"Saved results to {out_npz}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(sr_values, d1_scores, 'o-', label='degree 1')
    plt.plot(sr_values, d3_scores, 's--', label='degree 3')
    plt.xlabel('Spectral radius')
    plt.ylabel('Capacity')
    plt.title('Degree-1 and Degree-3 capacity vs spectral radius')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
