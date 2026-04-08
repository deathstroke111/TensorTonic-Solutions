import numpy as np
import math

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    added_batch_dim = False
    if Q.ndim == 2:
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
        added_batch_dim = True

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    batch_size, seq_len, d_model = Q_proj.shape
    d_k = d_model // num_heads

    Q_proj = Q_proj.reshape(batch_size, num_heads, seq_len, d_k)
    K_proj = K_proj.reshape(batch_size, num_heads, seq_len, d_k)
    V_proj = V_proj.reshape(batch_size, num_heads, seq_len, d_k)

    scores = np.matmul(Q_proj, np.transpose(K_proj, (0, 1, 3, 2))) / math.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    attention_output = np.matmul(attention_weights, V_proj)

    attention_output = attention_output.reshape(batch_size, seq_len, d_model)
    output = attention_output @ W_o

    if added_batch_dim:
        return output[0]
    return output