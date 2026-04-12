def apply_homogeneous_transform(T, points):
    T = np.asarray(T)
    points = np.asarray(points)

    # Ensure points are 2D: (N, 3) or (N, 4)
    if points.ndim == 1:
        points = points[None, :]  # make (1, D)

    # Add homogeneous coordinate if needed
    if points.shape[1] == 3:
        ones = np.ones((points.shape[0], 1))
        points = np.hstack([points, ones])

    # Apply transform
    transformed = (T @ points.T).T  # back to (N, 4)

    if len(transformed) == 1:
        return transformed[0][:3]
    
    return transformed[:, :3]
