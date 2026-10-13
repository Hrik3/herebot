"""
curiosity_tracker.py  –  Session-persistent Metrics Tracker for Herebot Curiosity Engine
==========================================================================================
Tracks, persists, and plots every key signal produced by CuriosityEngine across sessions.

Metrics recorded per interaction
─────────────────────────────────
  • words_known          – unique words in word_history
  • concepts_formed      – latent concept count  (concept_model)
  • total_reward         – cumulative intrinsic reward  (reward_tracker)
  • avg_entropy          – mean entropy across all known words
  • states_learned       – distinct interaction states  (state_manager)
  • top_curiosity_score  – curiosity score of the single most-curious word
  • most_curious_word    – the word that scored highest
  • factor_weights       – dict of meta-learner weights (one value per factor)
  • session_id           – UUID for the run that produced this point
  • interaction_num      – engine.interaction_counter at snapshot time
  • timestamp            – unix float

Persistence
───────────
  All data lives in  <graphs_dir>/tracker_data.json.
  Plots are written as PNGs alongside it.  On every save the graphs are
  regenerated, so opening the folder always shows the current view.

Usage (in CuriosityEngine or main())
──────────────────────────────────────
    from curiosity_tracker import CuriosityTracker

    tracker = CuriosityTracker()           # pass graphs_dir= to override default

    # inside process_input() or the main loop, after each interaction:
    tracker.record(engine)

    # optional – force a full plot refresh (done automatically on record):
    tracker.plot_all(engine)

    # on shutdown / long-idle:
    tracker.save_data()
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

# matplotlib is imported lazily inside plot_all() so the module can be imported
# even in headless environments without crashing at import time.

if TYPE_CHECKING:
    pass  # avoid circular import; CuriosityEngine is only used as a type hint


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default


# Gap in seconds between two consecutive data points that means "PC was off".
# Any pair of points further apart than this gets a None break inserted so
# matplotlib lifts the pen instead of drawing a flat line across the gap.
_SESSION_GAP_SECONDS: float = 300.0   # 5 minutes


def _break_series(timestamps, values, gap: float = _SESSION_GAP_SECONDS):
    """
    Insert a (midpoint_timestamp, None) pair between any two consecutive points
    whose timestamps differ by more than `gap` seconds.
    Works with both datetime objects and raw unix floats.
    Returns (new_timestamps, new_values) ready for ax.plot().
    Also retroactively fixes already-loaded historical data — no data is lost,
    only the visual connection between off-gap points is severed.
    """
    import datetime
    if not timestamps:
        return timestamps, values
    ts_out  = [timestamps[0]]
    val_out = [values[0]]
    for i in range(1, len(timestamps)):
        prev, curr = timestamps[i - 1], timestamps[i]
        if isinstance(prev, datetime.datetime):
            delta = (curr - prev).total_seconds()
            mid   = prev + (curr - prev) / 2
        else:
            delta = float(curr - prev)
            mid   = prev + delta / 2
        if delta > gap:
            ts_out.append(mid)
            val_out.append(None)        # pen-lift point
        ts_out.append(curr)
        val_out.append(values[i])
    return ts_out, val_out


# ──────────────────────────────────────────────────────────────────────────────
# CuriosityTracker
# ──────────────────────────────────────────────────────────────────────────────

class CuriosityTracker:
    """
    Lightweight, session-persistent metrics tracker for CuriosityEngine.

    Parameters
    ----------
    graphs_dir : str
        Directory where tracker_data.json and all PNG plots are stored.
        Created automatically if it does not exist.
    plot_every : int
        Re-render plots every N recorded interactions (default 5).
        Set to 1 to plot after every interaction (slower but always fresh).
    """

    DATA_FILE = "tracker_data.json"

    def __init__(self, graphs_dir: Optional[str] = None, plot_every: int = 5):
        # Keep tracker output inside the Herebot package regardless of the
        # shell's working directory (the application is normally launched
        # from ``herebot/`` but tools/tests may be launched elsewhere).
        if graphs_dir is None:
            graphs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graphs")
        self.graphs_dir = os.path.abspath(graphs_dir)
        self.plot_every = plot_every
        self._session_id: str = str(uuid.uuid4())[:8]
        self._records_since_last_plot: int = 0

        os.makedirs(self.graphs_dir, exist_ok=True)

        # List[dict] – chronological snapshots across ALL sessions
        self._data: List[Dict] = []
        self._load_data()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _data_path(self) -> str:
        return os.path.join(self.graphs_dir, self.DATA_FILE)

    def _load_data(self) -> None:
        path = self._data_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, list):
                    self._data = loaded
                    print(f"[Tracker] Loaded {len(self._data)} historical data points from {path}")
            except Exception as exc:
                print(f"[Tracker] Warning – could not load tracker data: {exc}")
                self._data = []

    def save_data(self) -> None:
        """Persist all records to JSON."""
        path = self._data_path()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
        except Exception as exc:
            print(f"[Tracker] Warning – could not save tracker data: {exc}")

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def record(self, engine, trigger: str = "interaction") -> None:
        """
        Snapshot the current engine state and append a data point.
        Automatically saves to JSON and (every `plot_every` calls) re-renders plots.

        trigger: 'interaction' | 'idle_short' | 'idle_long'
        """
        self._current_trigger = trigger
        snap = self._snapshot(engine)
        self._data.append(snap)
        self._records_since_last_plot += 1

        self.save_data()

        if self._records_since_last_plot >= self.plot_every:
            self.plot_all(engine)
            self._records_since_last_plot = 0

    def _snapshot(self, engine) -> Dict:
        """Extract every trackable metric from the engine right now."""
        trigger = getattr(self, "_current_trigger", "interaction")
        snap: Dict = {
            "schema_version": 2,
            "timestamp": time.time(),
            "session_id": self._session_id,
            "interaction_num": int(getattr(engine, "interaction_counter", 0)),
            "trigger": trigger,
        }

        # ── word / concept counts ────────────────────────────────────────────
        word_history = getattr(engine, "word_history", {})
        snap["words_known"] = len(word_history)

        concept_model = getattr(engine, "concept_model", None)
        if concept_model is not None:
            snap["concepts_formed"] = len(getattr(concept_model, "concept_counts", {}))
        else:
            snap["concepts_formed"] = 0

        # ── intrinsic reward ─────────────────────────────────────────────────
        reward_tracker = getattr(engine, "reward_tracker", None)
        if reward_tracker is not None:
            snap["total_reward"] = _safe_float(getattr(reward_tracker, "total_reward", 0.0))
            snap["episode_reward"] = _safe_float(getattr(reward_tracker, "episode_reward", 0.0))
        else:
            snap["total_reward"] = 0.0
            snap["episode_reward"] = 0.0

        # ── average entropy ──────────────────────────────────────────────────
        if concept_model is not None and word_history:
            try:
                entropies = [
                    _safe_float(concept_model.entropy_word(w))
                    for w in word_history
                ]
                snap["avg_entropy"] = float(np.mean(entropies)) if entropies else 0.0
            except Exception:
                snap["avg_entropy"] = 0.0
        else:
            snap["avg_entropy"] = 0.0

        # ── states learned ───────────────────────────────────────────────────
        state_manager = getattr(engine, "state_manager", None)
        if state_manager is not None:
            snap["states_learned"] = len(getattr(state_manager, "states", []))
        else:
            snap["states_learned"] = 0

        # ── top curiosity score & word ────────────────────────────────────────
        meta_learner = getattr(engine, "meta_learner", None)
        top_score = 0.0
        top_word = ""
        if meta_learner is not None and word_history:
            try:
                word_scores: Dict[str, float] = {}
                for w in word_history:
                    last = word_history[w][-1] if word_history[w] else {}
                    factors = last.get("curiosity_factors")
                    if factors is not None:
                        farr = engine._align_factor_vector(
                            factors.as_array(meta_learner.factor_names)
                        )
                        word_scores[w] = _safe_float(
                            meta_learner.predict_curiosity(farr)
                        )
                if word_scores:
                    top_word = max(word_scores, key=word_scores.get)
                    top_score = word_scores[top_word]
            except Exception:
                pass
        snap["top_curiosity_score"] = top_score
        snap["most_curious_word"] = top_word

        # ── meta-learner factor weights ───────────────────────────────────────
        if meta_learner is not None:
            try:
                weights = meta_learner.get_weights()          # returns dict str→float
                snap["factor_weights"] = {
                    k: _safe_float(v) for k, v in weights.items()
                }
            except Exception:
                snap["factor_weights"] = {}
        else:
            snap["factor_weights"] = {}

        # ── token profile aggregate ───────────────────────────────────────────
        token_profile = getattr(engine, "token_profile", {})
        snap["unique_tokens"] = len(token_profile)

        # ── GP predictive variance & uncertainty across vocabulary ───────────
        gp_model = getattr(engine, "gp_model", None)
        variances = []
        uncertainties = []
        if gp_model is not None and word_history:
            for w in list(word_history.keys())[:200]:      # cap for speed
                hist = word_history[w]
                if hist:
                    last = hist[-1]
                    features = last.get("features")
                    if features:
                        farr = engine._features_to_array(features)
                        var = gp_model.predict_variance(farr)
                        unc = gp_model.uncertainty(farr)
                        if var is not None and np.isfinite(var):
                            variances.append(float(var))
                        if unc is not None and np.isfinite(unc):
                            uncertainties.append(float(unc))
        snap["avg_gp_variance"] = float(np.mean(variances)) if variances else None
        snap["avg_gp_uncertainty"] = float(np.mean(uncertainties)) if uncertainties else None

        # ── turn-level telemetry ──────────────────────────────────────────────
        turn_telem = getattr(engine, "_last_turn_telemetry", {})
        snap["turn_gp_uncertainty"] = turn_telem.get("turn_gp_uncertainty", None)
        snap["turn_info_gain"] = turn_telem.get("turn_info_gain", None)
        snap["turn_bayesian_surprise"] = turn_telem.get("turn_bayesian_surprise", None)
        snap["turn_curiosity_score"] = turn_telem.get("turn_curiosity_score", None)
        snap["turn_prediction_error"] = getattr(engine, "_last_prediction_error", None)
        snap["total_info_gain"] = _safe_float(getattr(engine, "total_info_gain", 0.0))

        # ── control factors ───────────────────────────────────────────────────
        snap["control_factors"] = getattr(engine, "_last_control_factors", {})

        return snap

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_all(self, engine=None) -> None:
        """
        Render (or re-render) all graphs from accumulated data.
        Each graph is saved as a separate PNG in `graphs_dir`.
        """
        if not self._data:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")          # non-interactive backend; safe on any server
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("[Tracker] matplotlib not installed – skipping plots. "
                  "Run: pip install matplotlib")
            return

        # Sort data chronologically (may be out of order if loaded from file)
        data = sorted(self._data, key=lambda d: d.get("timestamp", 0))

        # Never draw a line across an engine restart.  The sessions with at
        # least ten snapshots form the readable trajectory panels; shorter
        # sessions remain represented in the session-count summary and all
        # sessions remain present in the heatmap.
        session_rows: Dict[str, List[Dict]] = defaultdict(list)
        for row in data:
            session_rows[row.get("session_id", "?")].append(row)
        substantive_sessions = [
            (session_id, rows) for session_id, rows in session_rows.items()
            if len(rows) >= 10
        ]

        # Build time axis as datetime objects for x-axis labels
        import datetime
        ts = [datetime.datetime.fromtimestamp(d["timestamp"]) for d in data]
        idx = list(range(len(data)))

        def _is_boundary(i: int) -> bool:
            """Return True at a real session change or a long recording gap."""
            if i <= 0:
                return False
            gap = (ts[i] - ts[i - 1]).total_seconds()
            return (
                data[i].get("session_id") != data[i - 1].get("session_id")
                or gap > _SESSION_GAP_SECONDS
            )

        def _bs(vals):
            """Sequential x-axis with a pen lift at each session boundary."""
            xs, ys = [], []
            for i, (x, y) in enumerate(zip(idx, vals)):
                if _is_boundary(i):
                    xs.append((idx[i - 1] + idx[i]) / 2.0)
                    ys.append(None)
                xs.append(x)
                ys.append(y)
            return xs, ys

        # ── helper: shade sessions ──
        def _shade_sessions(ax, data, ts):
            """Mark material recording gaps without obscuring dense sessions."""
            for i in range(1, len(data)):
                gap = (ts[i] - ts[i - 1]).total_seconds()
                if gap > _SESSION_GAP_SECONDS:
                    ax.axvline(
                        x=(idx[i - 1] + idx[i]) / 2.0,
                        color="#aaaaaa", linewidth=0.8,
                        linestyle="--", alpha=0.7, zorder=1,
                    )

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })

        def _plot_metric_by_session(ax, metric: str, title: str, ylabel: str,
                                    *, zero_is_missing: bool = False) -> None:
            """Draw comparable session-local traces, never a merged timeline."""
            for i, (session_id, rows) in enumerate(substantive_sessions):
                values = []
                for row in rows:
                    value = row.get(metric)
                    if value is None or (zero_is_missing and value == 0.0):
                        values.append(np.nan)
                    else:
                        values.append(value)
                ax.plot(range(1, len(rows) + 1), values,
                        color=_PALETTE[i % len(_PALETTE)], linewidth=1.5,
                        label=f"{session_id[:8]} (n={len(rows)})")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Snapshot within session", fontsize=7)

        # ──────────────────────────────────────────────────────────────────────
        # 1.  Overview dashboard  (3-row grid)
        # ──────────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Herebot Curiosity Engine – Session Tracker", fontsize=15, fontweight="bold")
        gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        # All dashboard traces use a session-local axis.  A global index would
        # turn state restoration between sessions into misleading jumps.
        # 1a. Vocabulary
        ax = fig.add_subplot(gs[0, 0])
        _plot_metric_by_session(ax, "words_known", "Vocabulary by Session", "Words known")
        ax.legend(fontsize=6, ncol=2, loc="best", title="Session", title_fontsize=7)

        # 1b. Intrinsic reward
        ax = fig.add_subplot(gs[0, 1])
        _plot_metric_by_session(ax, "total_reward", "Cumulative Reward by Session", "Reward")

        # 1c. Entropy & GP variance
        ax = fig.add_subplot(gs[1, 0])
        _plot_metric_by_session(ax, "avg_entropy", "Average Entropy by Session", "Entropy")

        # 1d. Top curiosity score
        ax = fig.add_subplot(gs[1, 1])
        _plot_metric_by_session(ax, "top_curiosity_score", "Top Curiosity Score by Session", "Score")

        # 1e. States & unique tokens
        ax = fig.add_subplot(gs[2, 0])
        _plot_metric_by_session(ax, "unique_tokens", "Unique Tokens by Session", "Tokens")

        # 1f. Session interaction distribution (bar chart)
        ax = fig.add_subplot(gs[2, 1])
        sessions: Dict[str, int] = defaultdict(int)
        for d in data:
            sessions[d.get("session_id", "?")[:6]] += 1
        # A horizontal ranking remains legible even with many short sessions.
        ordered_sessions = sorted(sessions.items(), key=lambda item: (-item[1], item[0]))
        sess_labels = [item[0] for item in ordered_sessions]
        sess_counts = [item[1] for item in ordered_sessions]
        bars = ax.barh(sess_labels, sess_counts,
                       color=[_PALETTE[i % len(_PALETTE)] for i in range(len(sess_labels))])
        ax.set_title("Recorded Snapshots per Session")
        ax.set_xlabel("Snapshots")
        ax.set_ylabel("Session ID (short)")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
        for bar, count in zip(bars, sess_counts):
            ax.text(bar.get_width() + max(sess_counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(count), ha="left", va="center", fontsize=7)

        self._savefig(fig, "overview_dashboard.png")
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────────
        # 2.  Factor weights heat-map  (rows = factors, cols = data points)
        # ──────────────────────────────────────────────────────────────────────
        all_factor_names: List[str] = []
        for d in data:
            for k in d.get("factor_weights", {}):
                if k not in all_factor_names:
                    all_factor_names.append(k)

        if all_factor_names:
            # A missing factor did not have a weight at that snapshot.  It is
            # unknown, not zero; preserving it as NaN prevents invented zero
            # bands in the heatmap and factor traces.
            matrix = np.array([
                [d.get("factor_weights", {}).get(f, np.nan) for f in all_factor_names]
                for d in data
            ], dtype=float).T                 # shape: (n_factors, n_snapshots)
            limit = float(np.nanmax(np.abs(matrix)))
            limit = max(limit, 1e-9)
            cmap = plt.get_cmap("RdYlGn").copy()
            cmap.set_bad(color="#e6e6e6")

            fig, ax = plt.subplots(figsize=(18, max(7, len(all_factor_names) * 0.34 + 2)))
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit,
                           interpolation="nearest")
            ax.set_yticks(range(len(all_factor_names)))
            ax.set_yticklabels(all_factor_names, fontsize=8)
            ax.set_xlabel("Chronological snapshot index")
            ax.set_title("Meta-Learner Factor Weights (grey = factor not yet present)")
            plt.colorbar(im, ax=ax, label="Weight", pad=0.01)
            fig.tight_layout()
            self._savefig(fig, "factor_weights_heatmap.png")
            plt.close(fig)

            # Factor weight trajectories are faceted by run.  Earlier plots
            # laid independent sessions end-to-end, which made resets appear
            # as random mid-series jumps.  Dynamic factors remain fully
            # visible in the heatmap; these panels focus on comparable core
            # factor trajectories within each actual engine run.
            core_factors = [f for f in all_factor_names if not (
                f.startswith("residual_") or f.startswith("drive_") or f.startswith("ctrl_")
            )]
            panel_sessions = substantive_sessions or list(session_rows.items())
            cols = 2
            rows = int(np.ceil(len(panel_sessions) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(18, max(6, rows * 3.4)),
                                     squeeze=False, sharey=True)
            flat_axes = axes.ravel()
            for panel, (session_id, session_data) in enumerate(panel_sessions):
                ax = flat_axes[panel]
                local_x = range(1, len(session_data) + 1)
                for i, fname in enumerate(core_factors):
                    vals = [d.get("factor_weights", {}).get(fname, np.nan) for d in session_data]
                    ax.plot(local_x, vals, label=fname, color=_PALETTE[i % len(_PALETTE)],
                            linewidth=1.25, alpha=0.9)
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
                ax.set_title(f"Session {session_id[:8]}  (n={len(session_data)})", loc="left", fontsize=10)
                ax.set_ylabel("Weight")
                ax.set_xlabel("Snapshot within this session")
            for ax in flat_axes[len(panel_sessions):]:
                ax.set_visible(False)
            handles, labels = flat_axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center right", fontsize=7, title="Core factors")
            fig.suptitle("Core Meta-Learner Weights Within Individual Sessions", fontsize=15, fontweight="bold")
            fig.subplots_adjust(left=0.06, right=0.82, bottom=0.07, top=0.91, hspace=0.36, wspace=0.18)
            self._savefig(fig, "factor_weights_lines.png")
            plt.close(fig)

        # ──────────────────────────────────────────────────────────────────────
        # 3.  Top curious words  (frequency bar)
        # ──────────────────────────────────────────────────────────────────────
        word_freq: Dict[str, int] = defaultdict(int)
        for d in data:
            w = d.get("most_curious_word", "")
            if w:
                word_freq[w] += 1

        if word_freq:
            top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:20]
            fig, ax = plt.subplots(figsize=(12, 4))
            bars = ax.barh(
                top_words[::-1],
                [word_freq[w] for w in top_words[::-1]],
                color=_PALETTE[0],
            )
            ax.set_xlabel("Times ranked as most-curious word")
            ax.set_title("Top 20 Most-Curious Words (all sessions)")
            for bar, word in zip(bars, top_words[::-1]):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(word_freq[word]), va="center", fontsize=8)
            fig.tight_layout()
            self._savefig(fig, "top_curious_words.png")
            plt.close(fig)

        # ──────────────────────────────────────────────────────────────────────
        # 4.  Per-session comparison (box-plot of observed reward changes)
        # ──────────────────────────────────────────────────────────────────────
        session_rewards: Dict[str, List[float]] = defaultdict(list)
        prev_reward: Dict[str, float] = {}
        for d in sorted(data, key=lambda x: x.get("timestamp", 0)):
            sid = d.get("session_id", "?")[:6]
            r = d.get("total_reward")
            if r is None:
                continue
            if sid in prev_reward:
                delta = r - prev_reward[sid]
                # Negative changes are valid observations and must not be
                # silently discarded from a research plot.
                session_rewards[sid].append(delta)
            prev_reward[sid] = r

        if session_rewards:
            fig, ax = plt.subplots(figsize=(max(6, len(session_rewards) * 1.5), 5))
            sess_ids = list(session_rewards.keys())
            bp = ax.boxplot(
                [session_rewards[s] for s in sess_ids],
                labels=sess_ids,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch, color in zip(bp["boxes"], _PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title("Cumulative-Reward Change per Recorded Event by Session")
            ax.set_ylabel("Δ cumulative intrinsic reward")
            ax.set_xlabel("Session ID (short)")
            fig.tight_layout()
            self._savefig(fig, "reward_per_session.png")
            plt.close(fig)

        # ──────────────────────────────────────────────────────────────────────
        # 5.  Idle vs Interaction metric comparison
        # ──────────────────────────────────────────────────────────────────────
        triggers = [d.get("trigger", "interaction") for d in data]
        idle_mask  = [t in ("idle_short", "idle_long") for t in triggers]
        inter_mask = [t == "interaction" for t in triggers]

        if any(idle_mask) and any(inter_mask):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle("Idle vs Interaction: Metric Comparison", fontweight="bold")
            for ax, metric, label in zip(
                axes,
                ["avg_entropy", "top_curiosity_score", "avg_gp_variance"],
                ["Avg Entropy", "Top Curiosity Score", "Avg GP Variance"],
            ):
                # Omit absent/non-finite measurements instead of converting
                # them to zero, which would fabricate a distributional mass.
                idle_vals = [float(d[metric]) for d, m in zip(data, idle_mask)
                             if m and d.get(metric) is not None and np.isfinite(d[metric])]
                inter_vals = [float(d[metric]) for d, m in zip(data, inter_mask)
                              if m and d.get(metric) is not None and np.isfinite(d[metric])]
                bp = ax.boxplot(
                    [inter_vals, idle_vals],
                    labels=["Interaction", "Idle"],
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                )
                bp["boxes"][0].set_facecolor(_PALETTE[0])
                bp["boxes"][1].set_facecolor(_PALETTE[2])
                for box in bp["boxes"]:
                    box.set_alpha(0.7)
                ax.set_title(f"{label}  (interaction n={len(inter_vals)}, idle n={len(idle_vals)})")
            fig.tight_layout()
            self._savefig(fig, "idle_vs_interaction.png")
            plt.close(fig)

        # ──────────────────────────────────────────────────────────────────────
        # 6.  Word co-occurrence / relation graph (top 20 curious words)
        # ──────────────────────────────────────────────────────────────────────
        # Build co-occurrence: two words co-occur if they were both top-curious
        # within a sliding window of 5 consecutive snapshots
        word_window: List[str] = [
            d.get("most_curious_word", "") for d in data if d.get("most_curious_word")
        ]
        cooccur: Dict[tuple, int] = defaultdict(int)
        window_size = 5
        for i in range(len(word_window)):
            window = word_window[i: i + window_size]
            seen = set()
            for w in window:
                if w and w not in seen:
                    for w2 in window:
                        if w2 and w2 != w and w2 not in seen:
                            pair = tuple(sorted([w, w2]))
                            cooccur[pair] += 1
                    seen.add(w)

        if cooccur:
            try:
                import networkx as nx
                top_pairs = sorted(cooccur, key=cooccur.get, reverse=True)
                G = nx.Graph()
                for (w1, w2) in top_pairs:
                    G.add_edge(w1, w2, weight=cooccur[(w1, w2)])

                fig, ax = plt.subplots(figsize=(13, 9))
                pos = nx.spring_layout(G, seed=42, k=1.8)
                weights = [G[u][v]["weight"] for u, v in G.edges()]
                max_w = max(weights) if weights else 1
                nx.draw_networkx_nodes(G, pos, node_size=600,
                                       node_color=_PALETTE[0], alpha=0.85, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                nx.draw_networkx_edges(G, pos,
                                       width=[1 + 3 * (w / max_w) for w in weights],
                                       alpha=0.5, edge_color=_PALETTE[3], ax=ax)
                ax.set_title("Word Co-Curiosity Graph (all pairs across all sessions)")
                ax.axis("off")
                fig.tight_layout()
                self._savefig(fig, "word_relation_graph.png")
                plt.close(fig)
            except ImportError:
                print("[Tracker] networkx not installed – skipping word graph. "
                      "Run: pip install networkx")

        print(f"[Tracker] Plots saved to  '{self.graphs_dir}/'  "
              f"({len(self._data)} total data points)")

    def _savefig(self, fig, filename: str) -> None:
        path = os.path.join(self.graphs_dir, filename)
        try:
            fig.savefig(path, dpi=140, bbox_inches="tight")
        except Exception as exc:
            print(f"[Tracker] Could not save {filename}: {exc}")

    # ── Convenience ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a one-line human-readable summary of accumulated stats."""
        if not self._data:
            return "[Tracker] No data recorded yet."
        last = self._data[-1]
        n_sessions = len({d.get("session_id") for d in self._data})
        return (
            f"[Tracker] {len(self._data)} snapshots across {n_sessions} session(s) | "
            f"words={last.get('words_known', 0)}  "
            f"concepts={last.get('concepts_formed', 0)}  "
            f"reward={last.get('total_reward', 0.0):.2f}  "
            f"entropy={last.get('avg_entropy', 0.0):.3f}  "
            f"states={last.get('states_learned', 0)}"
        )

    @property
    def session_id(self) -> str:
        return self._session_id
