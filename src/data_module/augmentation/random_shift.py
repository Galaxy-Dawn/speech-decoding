import torch


def random_shift(X, shift_ratio=0.2, method="bi"):
    max_shift = int(X.size(-1) * shift_ratio)
    X_shifted = torch.zeros_like(X)
    if method == "bi":
        n_steps = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if n_steps > 0:
            X_shifted[..., n_steps:] = X[..., :-n_steps]
        elif n_steps < 0:
            X_shifted[..., :n_steps] = X[..., -n_steps:]
        return X_shifted
    elif method == "forward":
        n_steps = torch.randint(1, max_shift + 1, (1,)).item()
        X_shifted[..., :-n_steps] = X[..., n_steps:]
        return X_shifted
    elif method == "backward":
        n_steps = torch.randint(1, max_shift + 1, (1,)).item()
        X_shifted[..., n_steps:] = X[..., :-n_steps]
        return X_shifted


def one_direction_shift(X, shift_ratio=0.2):
    max_shift = int(X.size(-1) * shift_ratio)
    X_shifted = torch.zeros_like(X)
    n_steps = torch.randint(1, max_shift + 1, (1,)).item()
    X_shifted[..., n_steps:] = X[..., :-n_steps]
    return X_shifted


def random_shift_seg(X, y, shift_ratio=0.2, method="bi"):
    max_shift = int(X.size(-1) * shift_ratio)
    X_shifted = torch.zeros_like(X)
    y_shifted = torch.zeros_like(y)  # Create a shifted copy of y

    if method == "bi":
        n_steps = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if n_steps > 0:
            X_shifted[..., n_steps:] = X[..., :-n_steps]
            y_shifted[..., n_steps:] = y[..., :-n_steps]  # Apply the same shift to y
        elif n_steps < 0:
            X_shifted[..., :n_steps] = X[..., -n_steps:]
            y_shifted[..., :n_steps] = y[..., -n_steps:]  # Apply the same shift to y
        return X_shifted, y_shifted

    elif method == "forward":
        n_steps = torch.randint(1, max_shift + 1, (1,)).item()
        X_shifted[..., :-n_steps] = X[..., n_steps:]
        y_shifted[..., :-n_steps] = y[..., n_steps:]  # Apply the same shift to y
        return X_shifted, y_shifted

    elif method == "backward":
        n_steps = torch.randint(1, max_shift + 1, (1,)).item()
        X_shifted[..., n_steps:] = X[..., :-n_steps]
        y_shifted[..., n_steps:] = y[..., :-n_steps]  # Apply the same shift to y
        return X_shifted, y_shifted