import math


EPSILON = 1e-9


def dynamic_horizon(interaction_index: int, base: float = 80.0, log_scale: float = 0.25) -> float:
    """Interaction-space horizon H(t)=base*(1+log_scale*log(1+t))."""
    t = max(0, int(interaction_index or 0))
    return float(base) * (1.0 + float(log_scale) * math.log1p(t))


def probability_tier(probability: float, tier_count: int = 5) -> int:
    p = max(0.0, min(1.0, float(probability or 0.0)))
    return min(max(int(tier_count) - 1, 0), int(int(tier_count) * p))


def ensure_probabilistic_stats(
    stats,
    interaction_index: int = 0,
    initial_usage: float = 0.0,
    initial_probability: float = 0.0,
):
    """Fill missing probabilistic convention fields on a mutable stats dict."""
    if stats is None:
        stats = {}
    t = int(interaction_index or 0)
    stats.setdefault("usage", float(initial_usage or 0.0))
    stats.setdefault("reward_trace", 0.0)
    stats.setdefault("last_interaction", t)
    stats.setdefault("last_decay_interaction", stats.get("last_interaction", t))
    stats.setdefault("probability", float(initial_probability or 0.0))
    stats.setdefault("tier", probability_tier(stats["probability"]))
    stats.setdefault("phase", "active")
    stats.setdefault("activation", EPSILON)
    stats.setdefault("archive_pressure", 0.0)
    return stats


def decay_rate(reward_trace: float, interaction_index: int, config: dict = None) -> float:
    cfg = config or {}
    horizon = dynamic_horizon(
        interaction_index,
        base=cfg.get("horizon_base", 80.0),
        log_scale=cfg.get("horizon_log_scale", 0.25),
    )
    scale = 1.0 - 0.5 * math.tanh(float(reward_trace or 0.0))
    scale = max(cfg.get("min_decay_scale", 0.25), min(cfg.get("max_decay_scale", 2.0), scale))
    return (math.log(2.0) / max(horizon, EPSILON)) * scale


def update_probabilistic_stats(
    stats,
    interaction_index: int,
    reward: float = 0.0,
    weight: float = 0.0,
    config: dict = None,
):
    """
    Apply interaction-space exponential decay and optional usage/reward evidence.

    last_interaction is the last actual evidence turn. last_decay_interaction is
    private bookkeeping so repeated reads do not apply the same decay twice.
    """
    cfg = config or {}
    t = int(interaction_index or 0)
    stats = ensure_probabilistic_stats(stats, t)
    last_decay = int(stats.get("last_decay_interaction", stats.get("last_interaction", t)) or 0)
    delta = max(0, t - last_decay)
    lam = decay_rate(stats.get("reward_trace", 0.0), t, cfg)
    decay = math.exp(-lam * delta)

    stats["usage"] = max(0.0, float(stats.get("usage", 0.0) or 0.0) * decay)
    stats["reward_trace"] = float(stats.get("reward_trace", 0.0) or 0.0) * decay
    stats["last_decay_interaction"] = t

    w = max(0.0, float(weight or 0.0))
    if w > 0.0:
        rt = math.tanh(float(reward or 0.0))
        stats["usage"] += w
        stats["reward_trace"] += w * rt
        stats["last_interaction"] = t

    reward_gate = (1.0 + math.tanh(float(stats.get("reward_trace", 0.0) or 0.0))) / 2.0
    stats["activation"] = max(cfg.get("epsilon", EPSILON), float(stats["usage"]) * reward_gate)
    return stats


def normalize_probability_field(
    stats_map,
    interaction_index: int,
    reward: float = 0.0,
    touched_key=None,
    touched_weight: float = 0.0,
    config: dict = None,
    archive: bool = True,
):
    """Decay a field of candidates, normalize active probabilities, and archive stale entries."""
    cfg = config or {}
    t = int(interaction_index or 0)
    for key, stats in list(stats_map.items()):
        update_probabilistic_stats(
            stats,
            t,
            reward=reward if key == touched_key else 0.0,
            weight=touched_weight if key == touched_key else 0.0,
            config=cfg,
        )

    active_items = [
        (key, stats)
        for key, stats in stats_map.items()
        if stats.get("phase", "active") != "archived"
    ]
    total = sum(max(cfg.get("epsilon", EPSILON), float(stats.get("activation", EPSILON))) for _, stats in active_items)
    if total <= 0.0 and active_items:
        total = float(len(active_items))

    for key, stats in stats_map.items():
        if stats.get("phase", "active") == "archived":
            stats["probability"] = 0.0
            stats["tier"] = 0
            continue
        activation = max(cfg.get("epsilon", EPSILON), float(stats.get("activation", EPSILON)))
        stats["probability"] = activation / total if total > 0.0 else 0.0
        stats["tier"] = probability_tier(stats["probability"], cfg.get("tier_count", 5))
        stats["archive_pressure"] = archive_pressure(stats, t, cfg)
        if archive and stats["archive_pressure"] >= cfg.get("archive_pressure_threshold", 0.98):
            last = int(stats.get("last_interaction", 0) or 0)
            if t - last >= dynamic_horizon(t, cfg.get("horizon_base", 80.0), cfg.get("horizon_log_scale", 0.25)):
                stats["phase"] = "archived"
                stats["probability"] = 0.0
                stats["tier"] = 0
    return stats_map


def archive_pressure(stats, interaction_index: int, config: dict = None) -> float:
    cfg = config or {}
    t = int(interaction_index or 0)
    last = int(stats.get("last_interaction", 0) or 0)
    delta = max(0, t - last)
    horizon = dynamic_horizon(t, cfg.get("horizon_base", 80.0), cfg.get("horizon_log_scale", 0.25))
    p = max(0.0, min(1.0, float(stats.get("probability", 0.0) or 0.0)))
    if float(stats.get("activation", EPSILON) or EPSILON) <= cfg.get("epsilon", EPSILON) * 10.0:
        p = 0.0
    q = float(stats.get("reward_trace", 0.0) or 0.0)
    return math.tanh(delta / max(horizon, EPSILON)) * (1.0 - p) * (1.0 - max(0.0, math.tanh(q)))
