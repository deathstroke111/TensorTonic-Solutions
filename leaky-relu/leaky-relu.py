import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    def apply_relu(ele):
        return ele if ele>0 else alpha*ele
    vec_func = np.vectorize(apply_relu)
    return vec_func(x)