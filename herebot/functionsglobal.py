from .importglobal import np
def safe_choice(a):
    """
    Robust replacement for np.random.choice that accepts
    lists, tuples, sets, dicts, Counters, numpy arrays,
    or nested structures and always samples safely.
    """
    if a is None:
        return None

    # Convert mappings / Counters to keys
    if hasattr(a, "keys"):
        a = list(a.keys())
    else:
        a = list(a)

    if len(a) == 0:
        return None

    idx = np.random.randint(len(a))
    return a[idx]
def align_vector(vec: np.ndarray, target_dim: int, fill: float = 0.0) -> np.ndarray:
    """Pad or truncate vector safely to target dimension."""
    if vec is None:
        return np.zeros(target_dim)

    if len(vec) == target_dim:
        return vec

    if len(vec) > target_dim:
        return vec[:target_dim]

    out = np.full(target_dim, fill)
    out[:len(vec)] = vec
    return out
def align_factor_vector(vec: np.ndarray, target_dim: int, fill: float = 0.5):
    """Ensure vector matches target dimensionality."""
    if vec is None:
        return np.full(target_dim, fill)

    vec = np.asarray(vec).reshape(-1)

    if len(vec) == target_dim:
        return vec

    if len(vec) < target_dim:
        return np.pad(vec, (0, target_dim - len(vec)), constant_values=fill)

    return vec[:target_dim]
