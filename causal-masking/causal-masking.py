import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    mask_mat = np.triu(np.ones_like(scores)*mask_value, k=1)
    
    lower_mat = np.tril(scores, k=0)

    return lower_mat+mask_mat
    