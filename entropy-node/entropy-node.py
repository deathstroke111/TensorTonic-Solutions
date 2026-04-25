import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    classes = np.unique(np.array(y), return_counts=True)[:][1]
    proportions = np.divide(classes, sum(classes))
    return -np.dot(np.log2(proportions), proportions)