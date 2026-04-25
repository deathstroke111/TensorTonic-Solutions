import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x_bar = np.array(x)
    p_bar = np.array(p)

    assert x_bar.shape[0] == p_bar.shape[0]
    
    if np.isclose(sum(p_bar), 1, atol=10^(-6)):
         return np.dot(x_bar, p_bar)
    else:
        raise ValueError("Probabilities don't add up to 1")