#!/usr/bin/env python3
"""Reconcile tracker interaction snapshots against saved raw episodes.

Only fields that have a direct, one-to-one raw source are eligible for repair:
``episode_reward`` and ``control_factors``.  Other tracker metrics are engine
aggregates and must never be guessed from an episode.  Every execution writes
an audit report; ``--apply`` writes only source-proven corrections.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path


MATCH_WINDOW_SECONDS = 2.0


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path,
                        default=root / "graphs" / "tracker_data.json")
    parser.add_argument("--episodes", type=Path,
                        default=root / "curiosity_state" / "episodes")
    parser.add_argument("--audit", type=Path,
                        default=root / "graphs" / "tracker_reconciliation_audit.json")
    parser.add_argument("--apply", action="store_true",
                        help="write direct source-proven corrections to tracker JSON")
    args = parser.parse_args()

    tracker = load_json(args.tracker)
    if not isinstance(tracker, list):
        raise ValueError("tracker data must be a JSON list")
    ordered = sorted(enumerate(tracker), key=lambda item: item[1].get("timestamp", 0))
    timestamps = [row["timestamp"] for _, row in ordered]

    episodes = []
    for path in sorted(args.episodes.glob("*.json")):
        try:
            row = load_json(path)
            if isinstance(row.get("timestamp"), (int, float)):
                episodes.append((float(row["timestamp"]), path.name, row))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    episodes.sort(key=lambda item: item[0])

    corrections = []
    verified = []
    unmatched = []
    used_snapshot_indices = set()
    for episode_timestamp, filename, episode in episodes:
        position = bisect.bisect_left(timestamps, episode_timestamp)
        candidates = [i for i in (position - 1, position) if 0 <= i < len(ordered)]
        if not candidates:
            unmatched.append({"episode": filename, "reason": "no tracker timestamps"})
            continue
        candidate = min(candidates, key=lambda i: abs(timestamps[i] - episode_timestamp))
        original_index, snapshot = ordered[candidate]
        offset = snapshot["timestamp"] - episode_timestamp
        if (abs(offset) > MATCH_WINDOW_SECONDS or snapshot.get("trigger") != "interaction"
                or original_index in used_snapshot_indices):
            unmatched.append({"episode": filename, "nearest_offset_seconds": offset})
            continue
        used_snapshot_indices.add(original_index)

        expected_reward = episode.get("intrinsic_reward")
        expected_factors = episode.get("factor_vector")
        changes = {}
        if isinstance(expected_reward, (int, float)) and snapshot.get("episode_reward") != expected_reward:
            changes["episode_reward"] = {"old": snapshot.get("episode_reward"), "new": expected_reward}
        if isinstance(expected_factors, dict) and snapshot.get("control_factors") != expected_factors:
            changes["control_factors"] = {"old": snapshot.get("control_factors"), "new": expected_factors}

        entry = {
            "snapshot_index": original_index,
            "episode": filename,
            "offset_seconds": offset,
            "fields": list(changes),
        }
        if changes:
            corrections.append({**entry, "changes": changes})
            if args.apply:
                for field, value in changes.items():
                    tracker[original_index][field] = value["new"]
        else:
            verified.append(entry)

    audit = {
        "schema_version": 1,
        "match_window_seconds": MATCH_WINDOW_SECONDS,
        "tracker_snapshots": len(tracker),
        "raw_episodes": len(episodes),
        "verified_interaction_snapshots": len(verified),
        "source_proven_corrections": len(corrections),
        "unmatched_episodes": len(unmatched),
        "corrections": corrections,
        "unmatched": unmatched,
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if args.apply and corrections:
        args.tracker.write_text(json.dumps(tracker, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in audit if key not in {"corrections", "unmatched"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
