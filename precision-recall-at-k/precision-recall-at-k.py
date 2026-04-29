def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    rec_set = set(recommended[:k])
    rel_set = set(relevant)

    return [len(rec_set.intersection(rel_set))/k, len(rec_set.intersection(rel_set))/len(rel_set)]
    