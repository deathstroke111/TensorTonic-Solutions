import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    dropout_pattern = 1 - rng.random(np.array(x).shape)
    safe_den = np.where(1-p==0, 0.0, 1-p)
    dropout_pattern = (dropout_pattern>p).astype(type(dropout_pattern))/safe_den
    scaled_dropout_pattern = np.multiply(x, dropout_pattern)

    return scaled_dropout_pattern, dropout_pattern