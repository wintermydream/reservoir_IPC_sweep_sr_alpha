import numpy as np
from fct_reservoir_state import fct_reservoir_state
from esn import ESN

np.set_printoptions(precision=6, suppress=True, linewidth=120)

def debug_fct_api(act_fct, input_weights, Wrec, input_sequence, steps, omega, warmup_len=0):
    print("\n--- fct_reservoir_state.py API ---")
    num_nodes = Wrec.shape[0]
    OrthWeights = np.eye(num_nodes)  # 不用正交项
    alpha = 1.0
    collect_len = steps
    states = fct_reservoir_state(
        act_fct,
        input_weights,
        Wrec,
        OrthWeights,
        input_sequence,
        collect_len,
        warmup_len,
        omega,
        alpha
    )  # (num_nodes, steps)
    for t in range(steps):
        print(f"[fct API] step {t} state:", states[:, t])
    return states

def debug_esn_api(act_fct, input_weights, Wrec, input_sequence, steps, omega, warmup_len=0):
    print("\n--- esn.py API ---")
    num_nodes = Wrec.shape[0]
    I2R = (omega * input_weights).T  # (1, N)
    R2R = Wrec.T  # (N, N)
    B2R = np.zeros((1, num_nodes))
    reservoir = ESN(I2R=I2R, R2R=R2R, B2R=B2R, nonlinearity=act_fct)
    total_steps = warmup_len + steps
    inputs = input_sequence[:total_steps].reshape(-1, 1)
    states = reservoir.Batch(inputs)  # (total_steps, num_nodes)
    # 只显示 warmup 后的状态
    for t in range(steps):
        print(f"[esn API] step {t} state:", states[warmup_len + t, :])
    return states[warmup_len:, :]

if __name__ == '__main__':
    num_nodes = 5
    steps = 5
    warmup_len = 500
    omega = 1.0
    spectrum_radius = 1.2
    activation_function = np.tanh
    np.random.seed(42)
    input_sequence = 2 * np.random.rand(steps+warmup_len, 1) - 1
    input_weights = 2 * np.random.rand(num_nodes, 1) - 1
    random_matrix_for_wrec = np.random.randn(num_nodes, num_nodes)
    eigs = np.linalg.eigvals(random_matrix_for_wrec)
    Wrec = spectrum_radius * random_matrix_for_wrec / np.sqrt(np.max(np.abs(eigs)))

    # fct_reservoir_state.py API
    debug_fct_api(activation_function, input_weights, Wrec, input_sequence, steps, omega, warmup_len)
    # esn.py API
    debug_esn_api(activation_function, input_weights, Wrec, input_sequence, steps, omega, warmup_len) 