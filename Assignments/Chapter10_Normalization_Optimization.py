import numpy as np


def normalize_input(X, mean=None, std=None):
    """
    Standardize input features using training statistics.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features.
    mean : array-like, optional
        Feature means calculated from the training data.
    std : array-like, optional
        Feature standard deviations from the training data.

    Returns
    -------
    X_normalized : ndarray
        Standardized input features.
    mean : ndarray
        Feature means.
    safe_std : ndarray
        Standard deviations with zeros replaced by 1.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array.")

    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std, or neither.")

    if mean is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    # Avoid division by zero for constant features.
    safe_std = np.where(std == 0, 1.0, std)

    X_normalized = (X - mean) / safe_std

    return X_normalized, mean, safe_std


def generate_mini_batches(X, Y, batch_size, seed=None):
    """
    Shuffle and divide paired features and labels into mini-batches.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features.
    Y : array-like, shape (n_samples,) or (n_samples, 1)
        Target labels.
    batch_size : int
        Maximum number of samples in each mini-batch.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mini_batches : list
        List of (X_batch, Y_batch) tuples.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2 or Y.ndim not in (1, 2):
        raise ValueError("X must be 2D and Y must be 1D or 2D.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must have the same number of samples."
        )

    if (
        not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])

    mini_batches = []

    # The final batch includes any remaining samples.
    for start in range(0, X.shape[0], batch_size):
        batch_indices = indices[start:start + batch_size]

        X_batch = X[batch_indices]
        Y_batch = Y[batch_indices]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # Example training data.
    X_train = rng.normal(loc=5, scale=10, size=(1000, 5))
    Y_train = rng.integers(0, 2, size=(1000, 1))

    # Calculate normalization statistics from training data.
    X_train_normalized, mu, sigma = normalize_input(X_train)

    # Apply the same statistics to validation data.
    X_validation = rng.normal(loc=5, scale=10, size=(200, 5))
    X_validation_normalized, _, _ = normalize_input(
        X_validation, mean=mu, std=sigma
    )

    batches = generate_mini_batches(
        X_train_normalized,
        Y_train,
        batch_size=32,
        seed=0
    )

    print(f"Total mini-batches: {len(batches)}")
    print(f"First batch X shape: {batches[0][0].shape}")
    print(f"Last batch X shape: {batches[-1][0].shape}")

    print(
        "Normalized training means:",
        np.round(X_train_normalized.mean(axis=0), 6)
    )
    print(
        "Normalized training standard deviations:",
        np.round(X_train_normalized.std(axis=0), 6)
    )

    # Mini-batch training structure:
    # Use a different seed each epoch to change the shuffle order.
    for epoch in range(3):
        epoch_batches = generate_mini_batches(
            X_train_normalized,
            Y_train,
            batch_size=32,
            seed=epoch
        )

        for X_batch, Y_batch in epoch_batches:
            # Perform forward propagation.
            # Calculate loss and gradients.
            # Update model parameters.
            pass