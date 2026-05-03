import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = sum([1 for yp, yt in zip(y_pred, y_true) if yp==yt])
    fp = sum([1 for yp, yt in zip(y_pred, y_true) if yp!=yt])

    return (2*tp)/((2*tp)+2*fp)
