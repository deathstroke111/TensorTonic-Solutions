import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    windows = sliding_window_view(x, (W.shape[2], W.shape[3]), axis=(2, 3))
    # windows shape: (N, C_in, H_out, W_out, KH, KW)

    y = np.einsum('nchwkl,ockl->nohw', windows, W)
    return y + b.reshape(1, -1, 1, 1)