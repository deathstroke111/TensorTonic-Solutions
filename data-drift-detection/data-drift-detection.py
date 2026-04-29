import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    ref_count = np.array(reference_counts)
    norm_ref_count = np.divide(ref_count, sum(ref_count))

    prod_count = np.array(production_counts)
    norm_prod_count = np.divide(prod_count, sum(prod_count))

    tvd = np.sum(np.abs(norm_ref_count - norm_prod_count))/2

    return {"score": tvd, "drift_detected": True if tvd>threshold else False}

    