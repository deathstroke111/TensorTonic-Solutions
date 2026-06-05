import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    mat_len = len(x.shape)
    
    if not mat_len in [3, 4]:
        raise ValueError('No channel provided')
        
    # Write code here
    x = np.array(x)
    h,w = mat_len-2, mat_len-1

    return np.mean(np.mean(x, axis=w), axis=h)
        