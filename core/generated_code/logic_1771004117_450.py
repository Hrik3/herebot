# AUTO‑GENERATED CODE
# Logic ID: logic_1771004117_450
# State label: basic knowledge
# Created: 2026-02-13 23:09:01

def run(factor_vector: list) -> bool:
    """
    Returns True if the current factor vector matches this state.
    """
    from numpy import dot, linalg
    prototype = [4.9761, 0.2028, 0.5504, 0.6412, 0.0489, 0.3720, 0.1500, 0.0000, 0.7000, 0.9442, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    if len(factor_vector) != len(prototype):
        return False
    sim = dot(factor_vector, prototype) / (linalg.norm(factor_vector) * linalg.norm(prototype) + 1e-12)
    return sim > 0.85
    return False
