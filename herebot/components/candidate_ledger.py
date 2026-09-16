"""
INVARIANTS (do not violate when modifying this file):

1. REPLAY IS NOT NEW EVIDENCE.
   An episode may be replayed arbitrarily many times across sleep cycles,
   but contributes evidence to a candidate association at most once,
   tracked via processed_episode_ids. Only genuinely new episodes (or an
   episode receiving new externally-grounded information, which would require
   a new episode_id) produce new evidence.

2. INTERNALLY GENERATED EVIDENCE CANNOT INDEPENDENTLY COMMIT.
   DREAMED, COUNTERFACTUAL, HYPOTHESIS, and REHEARSED evidence may support and
   reinforce a candidate, but consolidate() enforces a hard, volume-
   independent requirement of >= min_grounded_distinct_episodes distinct
   OBSERVED/USER_CONFIRMED episodes, and a minimum grounded-evidence fraction
   of total evidence, before any commit can occur -- regardless of how much
   ungrounded evidence has accumulated.

3. BROADCAST-WINNER STATUS IS METADATA, NOT A WEIGHT.
   was_broadcast_winner is recorded for later empirical analysis of whether
   GWT-style broadcast selection predicts useful learning better than mere
   co-activation. It must not multiply into raw_weight until that hypothesis
   has been tested and confirmed against real data.
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

EPSILON = 1e-9

GROUNDED_PROVENANCE = {"OBSERVED", "USER_CONFIRMED"}

PROVENANCE_WEIGHT = {
    "OBSERVED": 1.0,
    "USER_CONFIRMED": 1.0,
    "INFERRED": 0.5,
    "REHEARSED": 0.5,
    "HYPOTHESIS": 0.2,
    "DREAMED": 0.1,
    "COUNTERFACTUAL": 0.0,
}

DEFAULT_CFG = {
    "diversity_target": 3,
    "candidate_min_pmi": 0.15,
    "min_distinct_episodes": 3,
    "min_commit_margin": 0.25,
    "min_kl_surprise": 0.05,
    "min_grounded_distinct_episodes": 2,
    "min_grounded_evidence_fraction": 0.15,
    "candidate_decay": 0.9,
    "candidate_prune_floor": 1e-3,
    "candidate_max_age_cycles": 30,
}


@dataclass
class CandidateAssociation:
    token: str
    label: str
    episode_id: str
    timestamp: float
    provenance: str
    state_diversity: int
    pmi: float
    raw_weight: float
    was_broadcast_winner: bool = False
    cycles_survived: int = 0


class CandidateLedger:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        # key: (token, label) -> list[CandidateAssociation]
        self.candidates: Dict[Tuple[str, str], List[CandidateAssociation]] = defaultdict(list)
        # Episode IDs that have already contributed to candidate generation
        self.processed_episode_ids: Set[str] = set()
        # Audit history of all committed consolidation events
        self.commit_history: List[dict] = []

    # ---------- PMI / base-rate scan ----------
    @staticmethod
    def _tagged_episodes(episodic_memory, window: int):
        eps = episodic_memory.episodes[-window:]
        return [
            ep for ep in eps
            if "active_states_at_time" in ep and ep.get("active_states_at_time")
        ]

    def _build_stats(self, episodes, tokenize_fn):
        """
        Compute episode-level frequencies over the single consistent sample space: episodes.
        - P(token) = episodes containing token / total episodes
        - P(label) = episodes containing label / total episodes
        - P(token, label) = episodes containing both / total episodes
        """
        token_episode_counts = Counter()
        label_episode_counts = Counter()
        pair_episode_counts = Counter()
        n_eps = len(episodes)

        for ep in episodes:
            toks = set(tokenize_fn(ep.get("user_input", "")))
            states = set(ep.get("active_states_at_time", []))
            for t in toks:
                token_episode_counts[t] += 1
            for lbl in states:
                label_episode_counts[lbl] += 1
            for t in toks:
                for lbl in states:
                    pair_episode_counts[(t, lbl)] += 1

        return token_episode_counts, label_episode_counts, pair_episode_counts, max(n_eps, 1)

    def generate(self, episodic_memory, tokenize_fn, window: int = 100) -> int:
        """
        Inspect recent episodes, compute statistics over the full window,
        and turn ONLY previously unprocessed episodes into candidate evidence.
        """
        episodes = self._tagged_episodes(episodic_memory, window)
        if not episodes:
            return 0

        token_counts, label_counts, pair_counts, n_eps = self._build_stats(episodes, tokenize_fn)

        added = 0
        for ep in episodes:
            ep_id = ep.get("id") or f"ep_legacy_{ep.get('timestamp')}"
            # INVARIANT 1: REPLAY IS NOT NEW EVIDENCE
            if ep_id in self.processed_episode_ids:
                continue

            states = set(ep.get("active_states_at_time", []))
            diversity = len(states)
            provenance = ep.get("provenance", "OBSERVED")
            prov_weight = PROVENANCE_WEIGHT.get(provenance, 0.3)
            diversity_scale = min(1.0, diversity / max(1, self.cfg["diversity_target"]))
            if diversity_scale <= 0:
                self.processed_episode_ids.add(ep_id)
                continue

            broadcast_winner = ep.get("global_broadcast_state")
            toks = set(tokenize_fn(ep.get("user_input", "")))

            for t in toks:
                p_token = token_counts[t] / n_eps
                for lbl in states:
                    p_label = label_counts[lbl] / n_eps
                    p_joint = pair_counts[(t, lbl)] / n_eps
                    if p_token <= 0 or p_label <= 0 or p_joint <= 0:
                        continue
                    pmi = math.log((p_joint / (p_token * p_label)) + EPSILON)
                    if pmi < self.cfg["candidate_min_pmi"]:
                        continue

                    # INVARIANT 3: BROADCAST-WINNER STATUS IS METADATA, NOT A WEIGHT
                    raw_weight = pmi * prov_weight * diversity_scale
                    if raw_weight <= 0:
                        continue

                    was_winner = bool(broadcast_winner and lbl == broadcast_winner)
                    self.candidates[(t, lbl)].append(CandidateAssociation(
                        token=t,
                        label=lbl,
                        episode_id=ep_id,
                        timestamp=time.time(),
                        provenance=provenance,
                        state_diversity=diversity,
                        pmi=pmi,
                        raw_weight=raw_weight,
                        was_broadcast_winner=was_winner,
                    ))
                    added += 1

            self.processed_episode_ids.add(ep_id)

        return added

    # ---------- decay / prune ----------
    def age_and_prune(self):
        for key in list(self.candidates.keys()):
            kept = []
            for c in self.candidates[key]:
                c.raw_weight *= self.cfg["candidate_decay"]
                c.cycles_survived += 1
                if c.raw_weight < self.cfg["candidate_prune_floor"]:
                    continue
                if c.cycles_survived > self.cfg["candidate_max_age_cycles"]:
                    continue
                kept.append(c)
            if kept:
                self.candidates[key] = kept
            else:
                del self.candidates[key]

    # ---------- consolidation ----------
    def _evidence_by_label(self, token: str) -> Dict[str, dict]:
        out = {}
        for (t, lbl), items in self.candidates.items():
            if t != token or not items:
                continue
            grounded_items = [c for c in items if c.provenance in GROUNDED_PROVENANCE]
            out[lbl] = {
                "total_evidence": sum(c.raw_weight for c in items),
                "total_distinct_episodes": len({c.episode_id for c in items}),
                "grounded_evidence": sum(c.raw_weight for c in grounded_items),
                "grounded_distinct_episodes": len({c.episode_id for c in grounded_items}),
                "provenance_breakdown": Counter(c.provenance for c in items),
                "any_broadcast_winner": any(c.was_broadcast_winner for c in items),
                "items": items,
            }
        return out

    def consolidate(self, word_state_assoc, interaction_index: int = 0) -> List[tuple]:
        """
        Consolidate staged candidate evidence into durable WordState associations.
        Returns list of (token, label, total_evidence, margin, kl) committed.
        """
        committed = []
        tokens = {t for (t, _lbl) in self.candidates.keys()}

        for token in tokens:
            by_label = self._evidence_by_label(token)
            if not by_label:
                continue

            # Rank labels by total evidence
            ranked = sorted(
                by_label.items(),
                key=lambda kv: kv[1]["total_evidence"],
                reverse=True,
            )
            (top_label, top), *rest = ranked
            runner_up_evidence = rest[0][1]["total_evidence"] if rest else 0.0

            # INVARIANT 2: HARD GROUNDED-EVIDENCE GATE (Volume-independent)
            # Internally generated data (DREAMED, REHEARSED, COUNTERFACTUAL)
            # CANNOT independently commit without grounded evidence prerequisites.
            if top["grounded_distinct_episodes"] < self.cfg["min_grounded_distinct_episodes"]:
                continue

            grounded_frac = (
                (top["grounded_evidence"] / top["total_evidence"])
                if top["total_evidence"] > 0 else 0.0
            )
            if grounded_frac < self.cfg["min_grounded_evidence_fraction"]:
                continue

            # Gate: Total distinct episodes
            if top["total_distinct_episodes"] < self.cfg["min_distinct_episodes"]:
                continue

            # Gate: Winning margin over runner-up
            margin = top["total_evidence"] - runner_up_evidence
            if margin < self.cfg["min_commit_margin"]:
                continue

            # Gate: Information-theoretic belief change (KL divergence)
            kl = self._simulate_kl(word_state_assoc, token, top_label, top["total_evidence"], interaction_index)
            if kl < self.cfg["min_kl_surprise"]:
                continue

            # All gates cleared -> Commit to durable WordState
            confidence = min(1.0, top["total_evidence"])
            word_state_assoc.add_association(
                token,
                top_label,
                confidence=confidence,
                source="sleep_consolidated_evidence_gated",
                interaction_index=interaction_index,
            )

            # Record detailed audit event
            commit_event = {
                "token": token,
                "label": top_label,
                "total_evidence": top["total_evidence"],
                "grounded_evidence": top["grounded_evidence"],
                "grounded_distinct_episodes": top["grounded_distinct_episodes"],
                "total_distinct_episodes": top["total_distinct_episodes"],
                "margin": margin,
                "kl": kl,
                "provenance_breakdown": dict(top["provenance_breakdown"]),
                "was_broadcast_winner": top["any_broadcast_winner"],
                "interaction_index": int(interaction_index or 0),
                "timestamp": time.time(),
            }
            self.commit_history.append(commit_event)
            committed.append((token, top_label, top["total_evidence"], margin, kl))

            # Consume: remove committed candidates for this token/label pair
            del self.candidates[(token, top_label)]

        return committed

    @staticmethod
    def _simulate_kl(word_state_assoc, token: str, label: str, evidence: float, interaction_index: int) -> float:
        import copy
        from components.probabilistic_labeling import ensure_probabilistic_stats, normalize_probability_field

        prior_stats = word_state_assoc.word_label_stats.get(token, {})
        prior_probs = {k: float(v.get("probability", 0.0)) for k, v in prior_stats.items()}

        sim_stats = copy.deepcopy(dict(prior_stats))
        if label not in sim_stats:
            sim_stats[label] = {}
        ensure_probabilistic_stats(sim_stats[label], interaction_index=interaction_index)
        normalize_probability_field(
            sim_stats,
            interaction_index=interaction_index,
            reward=0.5,
            touched_key=label,
            touched_weight=evidence,
            archive=False,
        )
        post_probs = {k: float(v.get("probability", 0.0)) for k, v in sim_stats.items()}

        kl = 0.0
        for k, p_post in post_probs.items():
            p_prior = prior_probs.get(k, 1e-10)
            if p_post <= 0:
                continue
            kl += p_post * math.log(max(p_post, 1e-12) / max(p_prior, 1e-12))

        return max(0.0, min(10.0, kl))

    # ---------- persistence ----------
    def save(self, path: str):
        data = {
            "version": 1,
            "processed_episode_ids": list(self.processed_episode_ids),
            "commit_history": self.commit_history,
            "candidates": {
                f"{t}|||{lbl}": [asdict(c) for c in items]
                for (t, lbl), items in self.candidates.items()
            },
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            data = json.load(f)

        self.processed_episode_ids = set(data.get("processed_episode_ids", []))
        self.commit_history = list(data.get("commit_history", []))
        self.candidates = defaultdict(list)
        for key, items in data.get("candidates", {}).items():
            t, lbl = key.split("|||", 1)
            self.candidates[(t, lbl)] = [CandidateAssociation(**it) for it in items]
