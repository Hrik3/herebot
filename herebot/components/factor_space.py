"""Versioned, name-addressed factor-space primitives.

This module deliberately contains no global factor registry.  Each owner
(MetaCuriosityLearner and AdaptiveControlSpace) owns a separate FactorSpace.
Array indices are serialization details of one version; names are identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import time
import uuid
import numpy as np

OBSERVED = "observed"
ABSENT = "absent"
UNAVAILABLE = "unavailable"
VALID_STATUSES = {OBSERVED, ABSENT, UNAVAILABLE}


class FactorSpaceInvariantError(ValueError):
    """Raised when a positional vector cannot be proved to match its space."""


@dataclass(frozen=True)
class FactorSpaceSnapshot:
    space_id: str
    version: int
    factor_names: tuple[str, ...]
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.factor_names)}

    def as_dict(self) -> Dict[str, Any]:
        return {"space_id": self.space_id, "version": self.version,
                "factor_names": list(self.factor_names), "factor_metadata": self.metadata}


class FactorSpace:
    """A monotonic history of ordered factor-name schemas for one meaning."""
    def __init__(self, space_id: str, factor_names: Sequence[str], *, version: int = 0,
                 history: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 factor_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None):
        self.space_id = space_id
        self.version = int(version)
        self.factor_metadata = {k: dict(v) for k, v in (factor_metadata or {}).items()}
        self._history: Dict[int, FactorSpaceSnapshot] = {}
        if history:
            for key, value in history.items():
                v = int(value.get("version", key))
                self._history[v] = FactorSpaceSnapshot(
                    value.get("space_id", space_id), v,
                    tuple(value["factor_names"]), dict(value.get("factor_metadata", {})))
        self._validate_names(factor_names)
        current = self._history.get(self.version)
        if current is None:
            self._history[self.version] = FactorSpaceSnapshot(
                space_id, self.version, tuple(factor_names), self.factor_metadata)
        elif tuple(factor_names) != current.factor_names:
            raise FactorSpaceInvariantError(
                f"{space_id} v{self.version} names disagree with persisted schema")

    @staticmethod
    def _validate_names(names: Sequence[str]):
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise FactorSpaceInvariantError("factor names must be unique, non-empty strings")

    @property
    def factor_names(self) -> List[str]:
        return list(self.snapshot().factor_names)

    def snapshot(self, version: Optional[int] = None) -> FactorSpaceSnapshot:
        version = self.version if version is None else int(version)
        try:
            return self._history[version]
        except KeyError as exc:
            raise FactorSpaceInvariantError(f"unknown {self.space_id} factor-space version {version}") from exc

    def change(self, factor_names: Sequence[str], *, reason: str, metadata: Optional[Mapping[str, Mapping[str, Any]]] = None):
        self._validate_names(factor_names)
        if tuple(factor_names) == self.snapshot().factor_names:
            return self.snapshot()
        self.version += 1
        if metadata is not None:
            self.factor_metadata = {k: dict(v) for k, v in metadata.items()}
        self._history[self.version] = FactorSpaceSnapshot(
            self.space_id, self.version, tuple(factor_names), self.factor_metadata)
        return self.snapshot()

    def add(self, name: str, *, metadata: Optional[Mapping[str, Any]] = None):
        if name in self.factor_names:
            return self.snapshot()
        if metadata is not None:
            self.factor_metadata[name] = dict(metadata)
        return self.change(self.factor_names + [name], reason="factor_added")

    def retire(self, name: str):
        if name not in self.factor_names:
            raise FactorSpaceInvariantError(f"cannot retire missing factor {name!r}")
        return self.change([n for n in self.factor_names if n != name], reason="factor_retired")

    def as_dict(self) -> Dict[str, Any]:
        return {"space_id": self.space_id, "version": self.version,
                "factor_metadata": self.factor_metadata,
                "history": {str(v): s.as_dict() for v, s in self._history.items()}}

    def validate_vector(self, vector: Sequence[float], *, version: Optional[int] = None):
        snapshot = self.snapshot(version)
        if len(vector) != len(snapshot.factor_names):
            raise FactorSpaceInvariantError(
                f"{self.space_id} v{snapshot.version}: vector length={len(vector)}, "
                f"factor_names={len(snapshot.factor_names)}")


def observation_from_vector(vector: Sequence[float], snapshot: FactorSpaceSnapshot,
                            statuses: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    if len(vector) != len(snapshot.factor_names):
        raise FactorSpaceInvariantError(
            f"{snapshot.space_id} v{snapshot.version}: vector length={len(vector)}, "
            f"factor_names={len(snapshot.factor_names)}")
    status = {name: (statuses or {}).get(name, OBSERVED) for name in snapshot.factor_names}
    if not set(status.values()) <= VALID_STATUSES:
        raise FactorSpaceInvariantError("unknown factor observation status")
    return {"space_id": snapshot.space_id, "version": snapshot.version,
            "factor_names": list(snapshot.factor_names), "vector": list(map(float, vector)),
            "statuses": status}


def compare_observations(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Cosine comparison over names observed numerically in *both* records."""
    if a["space_id"] != b["space_id"]:
        raise FactorSpaceInvariantError(f"cannot compare {a['space_id']} with {b['space_id']}")
    def values(obs):
        names, vector = obs["factor_names"], obs["vector"]
        if len(names) != len(vector):
            raise FactorSpaceInvariantError("observation vector/names mismatch")
        return dict(zip(names, vector))
    av, bv = values(a), values(b)
    ast, bst = a.get("statuses", {}), b.get("statuses", {})
    shared = [n for n in a["factor_names"] if n in bv and ast.get(n, OBSERVED) == OBSERVED and bst.get(n, OBSERVED) == OBSERVED]
    excluded = sorted((set(av) | set(bv)) - set(shared))
    aa = np.array([av[n] for n in shared], dtype=float)
    bb = np.array([bv[n] for n in shared], dtype=float)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    similarity = 0.0 if denom <= 1e-12 else float(np.dot(aa, bb) / denom)
    return {"event_id": uuid.uuid4().hex, "timestamp": time.time(),
            "space_a": a["space_id"], "version_a": a["version"],
            "space_b": b["space_id"], "version_b": b["version"],
            "shared_factor_names": shared, "excluded_factor_names": excluded,
            "similarity": similarity, "comparison_policy": "shared_observed_factor_names_cosine"}


def project_observation(source: Mapping[str, Any], target: FactorSpaceSnapshot,
                        *, missing_policy: str = "unavailable") -> Dict[str, Any]:
    """Explicit projection. No default is invented unless caller declares one."""
    if source["space_id"] != target.space_id:
        raise FactorSpaceInvariantError("cross-space projection is forbidden")
    values = dict(zip(source["factor_names"], source["vector"]))
    statuses = source.get("statuses", {})
    out, out_status = [], {}
    for name in target.factor_names:
        if name in values:
            out.append(values[name]); out_status[name] = statuses.get(name, OBSERVED)
        elif missing_policy == "unavailable":
            out.append(None); out_status[name] = ABSENT
        elif missing_policy.startswith("default:"):
            out.append(float(missing_policy.split(":", 1)[1])); out_status[name] = ABSENT
        else:
            raise FactorSpaceInvariantError("projection needs missing_policy='unavailable' or 'default:<number>'")
    return {"event_id": uuid.uuid4().hex, "timestamp": time.time(),
            "source_space": source["space_id"], "source_version": source["version"],
            "target_space": target.space_id, "target_version": target.version,
            "source_factors": list(source["factor_names"]), "target_factors": list(target.factor_names),
            "shared_factors": [n for n in target.factor_names if n in values],
            "newly_introduced_factors": [n for n in target.factor_names if n not in values],
            "retired_factors": [n for n in source["factor_names"] if n not in target.factor_names],
            "policy_used": missing_policy, "vector": out, "statuses": out_status}
