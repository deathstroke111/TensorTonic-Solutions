import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here

    base = 1000.0

    def freq(x):
        return 1 / (base**(2*x/d_model))
    
    i = np.arange(d_model//2)

    freqencies = freq(i)

    token_position = np.arange(seq_length)

    angles = np.outer(token_position, freqencies)
    
    pe_vector = np.zeros((seq_length, d_model))

    pe_vector[:, 1::2] = np.cos(angles)
    
    if d_model%2 == 0:
        pe_vector[:, 0::2] = np.sin(angles)
        return pe_vector

    pe_vector[:, 0:-1:2] = np.sin(angles)
    pe_vector[:, -1] = np.sin(token_position.reshape(-1,1) * freq(d_model//2)).reshape(-1)

    return pe_vector