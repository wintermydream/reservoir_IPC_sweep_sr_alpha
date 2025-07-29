"""Compute and plot degree-1/3/5/7 Legendre memory capacities versus input scale
alpha (\u03B1) with fixed spectral radius sr=1.0.

Results are cached in `debug_capacity_data/` so subsequent runs skip heavy
computations and directly plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
import capacity_calculation


def compute_deg_capacity(states: np.ndarray, inputs: np.ndarray, degree: int) -> float:
    """Aggregate capacity score for a single Legendre degree."""
    if states.shape[0] != inputs.shape[0]:
        m = min(states.shape[0], inputs.shape[0])
        states = states[-m:, :]
        inputs = inputs[-m:]
    cap_iter = capacity_calculation.CapacityIterator(mindeg=degree, maxdeg=degree, m_degrees=True)
    _, caps, _, _ = cap_iter.collect(inputs, states)
    return float(sum(c['score'] for c in caps if c['degree'] == degree))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute degree-1/3/5/7 capacity vs alpha (input scaling)")
    parser.add_argument('--data_dir', default='reservoir_states_data', help='directory with res_*.npz files')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--spectrum_radius', type=float, default=1.4)
    parser.add_argument('--alpha_len', type=int, default=30)
    parser.add_argument('--alpha_min', type=float, default=0.01)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--omega', type=float, default=1.0)
    parser.add_argument('--out_dir', default='debug_capacity_data')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"d1_d3_d5_d7_capacity_vs_alpha_N{args.num_nodes}_sr{args.spectrum_radius:.3f}_om{args.omega:.3f}.npz"

    degrees = [1, 3, 5, 7]

    if cache.exists():
        print(f"[Info] Cached data {cache.name} found – loading …")
        data = np.load(cache)
        alpha_values = data['alpha_values']
        d1_scores = data['d1_scores']
        d3_scores = data['d3_scores']
        d5_scores = data['d5_scores']
        d7_scores = data['d7_scores']
    else:
        alpha_values = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.alpha_len)
        d1_scores: list[float] = []
        d3_scores: list[float] = []
        d5_scores: list[float] = []
        d7_scores: list[float] = []

        data_dir = Path(args.data_dir)
        for alpha in alpha_values:
            fname = f"res_N{args.num_nodes}_sr{args.spectrum_radius:.3f}_om{args.omega:.3f}_al{alpha:.3f}.npz"
            fpath = data_dir / fname
            if not fpath.exists():
                print(f"[WARN] missing {fname}, skipping …")
                continue
            loaded = np.load(fpath)
            inputs: np.ndarray = loaded['input_sequence']
            states: np.ndarray = loaded['reservoir_state_memory']
            if states.shape[0] == args.num_nodes:
                states = states.T

            t0 = time.time()
            # Ensure shapes: states (T, N), inputs (T,)
            if states.shape[0] != inputs.shape[0]:
                min_len = min(states.shape[0], inputs.shape[0])
                states = states[-min_len:, :]
                inputs = inputs[-min_len:]
            
            cap_iter = capacity_calculation.CapacityIterator(mindeg=1, maxdeg=7, m_degrees=True)
            _, all_caps, _, _ = cap_iter.collect(inputs, states)
            d1_scores.append(sum(c['score'] for c in all_caps if c['degree'] == 1))
            d3_scores.append(sum(c['score'] for c in all_caps if c['degree'] == 3))
            d5_scores.append(sum(c['score'] for c in all_caps if c['degree'] == 5))
            d7_scores.append(sum(c['score'] for c in all_caps if c['degree'] == 7))
            print(f"alpha={alpha:.3f}: d1={d1_scores[-1]:.3f}, d3={d3_scores[-1]:.3f}, d5={d5_scores[-1]:.3f}, d7={d7_scores[-1]:.3f} (elapsed {time.time()-t0:.2f}s)")

        # Convert to numpy arrays for saving/plotting
        alpha_values = alpha_values[:len(d1_scores)]
        d1_scores = np.array(d1_scores)
        d3_scores = np.array(d3_scores)
        d5_scores = np.array(d5_scores)
        d7_scores = np.array(d7_scores)

        np.savez(cache, alpha_values=alpha_values, d1_scores=d1_scores, d3_scores=d3_scores, d5_scores=d5_scores,
                 d7_scores=d7_scores, num_nodes=args.num_nodes, spectrum_radius=args.spectrum_radius, omega=args.omega)
        print(f"Saved results to {cache}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.semilogx(alpha_values, d1_scores, 'o-', label='degree 1')
    plt.semilogx(alpha_values, d3_scores, 's--', label='degree 3')
    plt.semilogx(alpha_values, d5_scores, 'd:', label='degree 5')
    plt.semilogx(alpha_values, d7_scores, '^-', label='degree 7')
    plt.xlabel('Input scaling alpha (log scale)')
    plt.ylabel('Capacity')
    plt.title('Degree capacities vs alpha (sr=1)')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
