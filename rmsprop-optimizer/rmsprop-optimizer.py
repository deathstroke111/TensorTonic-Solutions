import numpy as np
import math

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    s_t = np.multiply(beta, s) + np.multiply((1-beta), np.power(g,2))
    w -= np.divide(np.multiply(lr, g), np.sqrt(s_t + eps))

    return w, s_t

    