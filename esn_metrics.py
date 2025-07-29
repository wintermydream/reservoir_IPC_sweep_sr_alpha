import numpy as np

def compute_memory_capacity(reservoir_states, input_sequence, maxMemory, trainLen, initLen):
    """
    Compute memory capacity (MC), consistent with MATLAB fct_shortMemory.m.
    Args:
        reservoir_states: (nDims, trainLen)
        input_sequence: (warmup_len + collect_len,)
        maxMemory: int
        trainLen: int
        initLen: int
    Returns:
        MC: (maxMemory,) memory capacity for each delay
    """
    pastInputTrainMtx = np.zeros((maxMemory, trainLen))
    for i in range(maxMemory):
        pastInputTrainMtx[i, :] = input_sequence[initLen - i : initLen - i + trainLen]
    outputWeights = pastInputTrainMtx @ np.linalg.pinv(reservoir_states)
    trainOutput = outputWeights @ reservoir_states
    corrmat = np.corrcoef(trainOutput, pastInputTrainMtx)
    MC = np.diag(corrmat[0:maxMemory, maxMemory:]) ** 2
    return MC 

def fct_reservoir_state(
    act_fct,
    input_weights,
    Wrec,
    OrthWeights,
    input_sequence,
    collect_len,
    warmup_len,
    omega,
    alpha,
    sigma_n=0
):
    """
    Simulate ESN reservoir and return reservoir_state_memory (collected states after warmup).

    Parameters:
        act_fct: activation function, e.g., np.tanh
        input_weights: input weight vector, shape (num_nodes, 1)
        Wrec: recurrent weight matrix, shape (num_nodes, num_nodes)
        OrthWeights: orthogonal weight matrix, shape (num_nodes, num_nodes)
        input_sequence: input array, shape (total_steps,) or (total_steps, 1)
        collect_len: number of steps to record reservoir states after warmup
        warmup_len: number of warmup steps
        omega: input scaling factor (float)
        alpha: mixing parameter (float)
        sigma_n: noise standard deviation (float)
    Returns:
        reservoir_state_memory: array of shape (num_nodes, collect_len)
    """
    num_nodes = Wrec.shape[0]
    total_steps = warmup_len + collect_len
    input_sequence = np.asarray(input_sequence).reshape(-1, 1)
    assert input_sequence.shape[0] >= total_steps, "input_sequence is too short for warmup_len + collect_len"

    reservoir_state = np.zeros((num_nodes, 1))
    reservoir_state_memory = np.zeros((num_nodes, collect_len))

    # Warm up the reservoir
    for t in range(warmup_len):
        input_vector = omega * input_weights[:, 0] * input_sequence[t, 0]  # (num_nodes,)
        input_vector = input_vector.reshape(-1, 1)  # 保持后续 shape 一致
        reservoir_state = alpha * act_fct(input_vector + Wrec @ reservoir_state + sigma_n * np.random.randn(num_nodes, 1)) \
            + (1 - alpha) * (OrthWeights @ reservoir_state)

    # Collect reservoir states
    for t in range(collect_len):
        input_vector = omega * input_weights[:, 0] * input_sequence[warmup_len + t, 0]  # (num_nodes,)
        input_vector = input_vector.reshape(-1, 1)
        reservoir_state = alpha * act_fct(input_vector + Wrec @ reservoir_state + sigma_n * np.random.randn(num_nodes, 1)) \
            + (1 - alpha) * (OrthWeights @ reservoir_state)
        reservoir_state_memory[:, t] = reservoir_state[:, 0]

    return reservoir_state_memory


def fct_memory_capacity(
    reservoir_states: np.ndarray,
    input_sequence: np.ndarray,
    maxMemory: int,
    trainLen: int,
    initLen: int,
    normalize_target: bool = False,
) -> np.ndarray:
    """Compute 1st-degree memory capacity.

    Parameters mirror ``compute_memory_capacity`` with an optional
    ``normalize_target`` flag.
    """
    # Prepare past input matrix
    pastInputTrainMtx = np.zeros((maxMemory, trainLen))
    for k in range(maxMemory):
        tgt = input_sequence[initLen - k : initLen - k + trainLen]
        if normalize_target:
            mean = tgt.mean()
            std = tgt.std()
            tgt = (tgt - mean) / std if std > 0 else (tgt - mean)
        pastInputTrainMtx[k, :] = tgt

    # Train linear readout
    outputWeights = pastInputTrainMtx @ np.linalg.pinv(reservoir_states)
    trainOutput = outputWeights @ reservoir_states

    # R^2 for each delay
    corrmat = np.corrcoef(trainOutput, pastInputTrainMtx)
    mc = np.diag(corrmat[:maxMemory, maxMemory:]) ** 2
    return mc


def fct_d1_capacity(reservoir_states, input_sequence, maxMemory, trainLen, initLen):
    """Compute first-degree memory capacity (Legendre degree 1) using the
    covariance method from `capacity_calculation.py`.

    This has the *same* calling interface as ``compute_memory_capacity`` so that
    it can be used as a drop-in replacement.

    Parameters
    ----------
    reservoir_states : ndarray, shape (nDims, trainLen)
        Collected reservoir state matrix (each column is a time step).
    input_sequence : ndarray, shape (warmup_len + trainLen,)
        Warm-up + training input sequence that generated the states.
    maxMemory : int
        Maximum delay to evaluate.
    trainLen : int
        Number of training samples/columns in ``reservoir_states``.
    initLen : int
        Index of first training sample in ``input_sequence``.

    Returns
    -------
    mc : ndarray, shape (maxMemory,)
        Degree-1 memory capacity for each delay.
    """
    # States as (samples, dims)
    states = reservoir_states.T  # shape (trainLen, nDims)
    N = trainLen

    # Inverse of state covariance (regularised via pseudoinverse)
    R_inv = np.linalg.pinv(states.T @ states / N)

    mc = np.zeros(maxMemory)
    for k in range(maxMemory):
        target = input_sequence[initLen - k : initLen - k + trainLen].reshape(-1, 1)
        # Cross-covariance between states and target
        P = states.T @ target / N  # shape (nDims, 1)
        # Capacity score following `cov_capacity`
        mc[k] = float((P.T @ R_inv @ P) / np.mean(target ** 2))
    return mc.flatten()