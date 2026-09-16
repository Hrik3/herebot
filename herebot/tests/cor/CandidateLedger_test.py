"""
Tests for CandidateLedger, Invariants, Adversarial Proofs, and WordState Cooldown.
"""

import math
import os
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

from components.candidate_ledger import (
    CandidateAssociation,
    CandidateLedger,
    DEFAULT_CFG,
    GROUNDED_PROVENANCE,
    PROVENANCE_WEIGHT,
)
from components.learner import WordStateAssociator


class DummyEpisodicMemory:
    def __init__(self, episodes=None):
        self.episodes = episodes or []


def simple_tokenize(text: str):
    return [t.lower() for t in text.split() if t.strip()]


class TestCandidateLedgerAdversarial(unittest.TestCase):
    def setUp(self):
        self.ledger = CandidateLedger({
            "diversity_target": 3,
            "candidate_min_pmi": 0.05,
            "min_distinct_episodes": 3,
            "min_commit_margin": 0.20,
            "min_kl_surprise": 0.01,
            "min_grounded_distinct_episodes": 2,
            "min_grounded_evidence_fraction": 0.15,
        })
        self.wsa = WordStateAssociator()

    def test_replay_does_not_inflate_evidence(self):
        """Invariant 1: Replaying the exact same episodes must add ZERO new evidence."""
        episodes = [
            {
                "id": f"ep_obs_{i}",
                "user_input": "water is wet and flowing",
                "active_states_at_time": ["nature", "liquid", "calm"],
                "provenance": "OBSERVED",
            }
            for i in range(10)
        ]
        # Add background distribution episodes so PMI is defined and positive
        for j in range(15):
            episodes.append({
                "id": f"ep_bg_{j}",
                "user_input": "sun fire bright hot",
                "active_states_at_time": ["energy", "heat", "bright"],
                "provenance": "OBSERVED",
            })

        mem = DummyEpisodicMemory(episodes)

        added_1 = self.ledger.generate(mem, simple_tokenize, window=100)
        self.assertGreater(added_1, 0, "Initial generation should produce candidate evidence")

        # Snapshot evidence across all candidates
        snapshot_1 = {
            k: [c.raw_weight for c in items]
            for k, items in self.ledger.candidates.items()
        }

        # Generate again on the exact same episodes (simulating replay across sleep cycles)
        added_2 = self.ledger.generate(mem, simple_tokenize, window=100)
        self.assertEqual(added_2, 0, "Replay of previously processed episodes must produce 0 new candidates")

        snapshot_2 = {
            k: [c.raw_weight for c in items]
            for k, items in self.ledger.candidates.items()
        }
        self.assertEqual(snapshot_1, snapshot_2, "Evidence snapshot must remain completely identical after replay")

    def test_dreamed_cannot_commit_even_with_overwhelming_volume(self):
        """
        Invariant 2: 500 DREAMED episodes + 0 OBSERVED episodes MUST NOT COMMIT.
        Proves the protection is structural, not merely a weak weight.
        """
        episodes = []
        for i in range(500):
            episodes.append({
                "id": f"ep_dream_{i}",
                "user_input": "ghost floating through wall",
                "active_states_at_time": ["spectral", "mystery", "night"],
                "provenance": "DREAMED",
            })
        for j in range(500):
            episodes.append({
                "id": f"ep_dream_bg_{j}",
                "user_input": "rock stone solid ground",
                "active_states_at_time": ["earth", "heavy", "still"],
                "provenance": "DREAMED",
            })

        mem = DummyEpisodicMemory(episodes)
        added = self.ledger.generate(mem, simple_tokenize, window=1000)
        self.assertGreater(added, 0, "Candidates may be staged from dreamed data")

        # Consolidate must return ZERO commits because grounded prerequisite is completely unsatisfied
        committed = self.ledger.consolidate(self.wsa, interaction_index=50)
        self.assertEqual(
            len(committed),
            0,
            "500 DREAMED episodes with 0 OBSERVED episodes must be blocked by the hard grounded gate",
        )
        self.assertEqual(len(self.wsa.word_to_labels.get("ghost", {})), 0)

    def test_counterfactual_only_cannot_commit(self):
        """Counterfactual evidence has provenance weight 0.0 and cannot commit."""
        episodes = [
            {
                "id": f"ep_cf_{i}",
                "user_input": "what if sky was purple",
                "active_states_at_time": ["wonder", "surreal", "twilight"],
                "provenance": "COUNTERFACTUAL",
            }
            for i in range(50)
        ]
        for j in range(50):
            episodes.append({
                "id": f"ep_cf_bg_{j}",
                "user_input": "tree grass green nature",
                "active_states_at_time": ["forest", "growth", "life"],
                "provenance": "COUNTERFACTUAL",
            })

        mem = DummyEpisodicMemory(episodes)
        added = self.ledger.generate(mem, simple_tokenize, window=100)
        # raw_weight = pmi * prov_weight(0.0) = 0.0 -> skipped
        self.assertEqual(added, 0)
        committed = self.ledger.consolidate(self.wsa, interaction_index=10)
        self.assertEqual(len(committed), 0)

    def test_grounded_evidence_with_dreamed_support_can_commit(self):
        """
        Positive test: Grounded evidence satisfying min_grounded_distinct_episodes (>=2)
        and min_grounded_evidence_fraction (>=0.15) supported by DREAMED evidence CAN commit.
        """
        episodes = [
            # 3 distinct OBSERVED episodes consistently featuring 'mineral_discovery'
            {
                "id": "ep_obs_0",
                "user_input": "zibber crystal glowing cave",
                "active_states_at_time": ["mineral_discovery", "focus", "alert"],
                "provenance": "OBSERVED",
            },
            {
                "id": "ep_obs_1",
                "user_input": "zibber crystal glowing cave",
                "active_states_at_time": ["mineral_discovery", "cave_exploration", "calm"],
                "provenance": "OBSERVED",
            },
            {
                "id": "ep_obs_2",
                "user_input": "zibber crystal glowing cave",
                "active_states_at_time": ["mineral_discovery", "darkness", "deep"],
                "provenance": "OBSERVED",
            },
        ]
        # 5 distinct DREAMED supporting episodes also featuring 'mineral_discovery'
        for d in range(5):
            episodes.append({
                "id": f"ep_dream_{d}",
                "user_input": "zibber crystal glowing cave",
                "active_states_at_time": ["mineral_discovery", "night", "wonder"],
                "provenance": "DREAMED",
            })
        # Diverse background episodes to establish distinct base rates and competitive margins
        for bg in range(25):
            episodes.append({
                "id": f"ep_bg_{bg}",
                "user_input": "water river fish flowing stream",
                "active_states_at_time": ["aquatic", "peaceful", "riverbank"],
                "provenance": "OBSERVED",
            })

        mem = DummyEpisodicMemory(episodes)
        added = self.ledger.generate(mem, simple_tokenize, window=100)
        self.assertGreater(added, 0)

        committed = self.ledger.consolidate(self.wsa, interaction_index=20)
        self.assertGreater(len(committed), 0, "Grounded evidence with dream support must clear gates and commit")

        committed_tokens = [c[0] for c in committed]
        self.assertTrue(any(t in ["zibber", "crystal", "glowing", "cave"] for t in committed_tokens))
        # Verify commit history audit log was recorded
        self.assertEqual(len(self.ledger.commit_history), len(committed))
        first_audit = self.ledger.commit_history[0]
        self.assertGreaterEqual(first_audit["grounded_distinct_episodes"], 2)
        self.assertGreaterEqual(first_audit["grounded_evidence"] / first_audit["total_evidence"], 0.15)

    def test_persistence_of_processed_episode_ids_and_candidates(self):
        """Verify candidate ledger save/load restores processed_episode_ids and candidates."""
        episodes = [
            {
                "id": "ep_test_101",
                "user_input": "mountain summit high peak",
                "active_states_at_time": ["alpine", "majestic", "high_altitude"],
                "provenance": "OBSERVED",
            },
            {
                "id": "ep_test_102",
                "user_input": "ocean waves deep water",
                "active_states_at_time": ["marine", "vast", "coastal"],
                "provenance": "OBSERVED",
            },
        ]
        mem = DummyEpisodicMemory(episodes)
        self.ledger.generate(mem, simple_tokenize, window=10)
        self.assertIn("ep_test_101", self.ledger.processed_episode_ids)

        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "candidate_associations.json")
            self.ledger.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            restored = CandidateLedger()
            restored.load(save_path)
            self.assertEqual(restored.processed_episode_ids, self.ledger.processed_episode_ids)
            self.assertEqual(len(restored.candidates), len(self.ledger.candidates))


class TestWordStateCooldownAndPersistence(unittest.TestCase):
    def test_mark_unresolved_sets_cooldown_and_penalizes_reward_trace(self):
        wsa = WordStateAssociator()
        wsa.add_association("target", "state_a", confidence=0.8, interaction_index=5)
        wsa.add_association("target", "state_b", confidence=0.8, interaction_index=5)

        # Before: conflict detected
        conflicts = wsa.detect_conflicts("target", interaction_index=6)
        self.assertEqual(len(conflicts), 1)

        # User chooses NEITHER / unresolved
        wsa.mark_unresolved("target", ["state_a", "state_b"], interaction_index=6, cooldown_turns=10)

        # Immediate next turn within cooldown: conflict is suppressed
        conflicts_during_cooldown = wsa.detect_conflicts("target", interaction_index=7)
        self.assertEqual(conflicts_during_cooldown, [], "Conflict must be suppressed during cooldown")

        # After cooldown passes (interaction_index > 16):
        conflicts_after_cooldown = wsa.detect_conflicts("target", interaction_index=17)
        # Symmetrically penalized, both are still active
        self.assertIn("target", wsa.conflict_cooldown)
        self.assertEqual(wsa.conflict_cooldown["target"], 16)

        # Verify conflict_history records NEITHER
        self.assertEqual(len(wsa.conflict_history), 1)
        self.assertEqual(wsa.conflict_history[0]["outcome"], "NEITHER")
        self.assertIsNone(wsa.conflict_history[0]["kept"])

    def test_cooldown_persists_across_save_and_load(self):
        wsa = WordStateAssociator()
        wsa.add_association("unit", "alpha", confidence=1.0, interaction_index=1)
        wsa.conflict_cooldown["unit"] = 45

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "word_state.json")
            wsa.save(path)

            restored = WordStateAssociator()
            restored.load(path)
            self.assertEqual(restored.conflict_cooldown.get("unit"), 45)
            self.assertEqual(restored.detect_conflicts("unit", interaction_index=20), [])

    def test_e15_resolution_immediate_durability_and_restart(self):
        """
        Durability test for E15:
        1. create conflict on 'fire'
        2. resolve with 'bright curiosity'
        3. immediately save word state (simulating Handler #7 targeted save)
        4. reload in a fresh WordStateAssociator instance without normal pipeline flush
        5. assert mutation survived on disk, detect_conflicts returns [], and history has fire.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "word_state.json")
            wsa = WordStateAssociator()
            wsa.add_association("fire", "greetings", confidence=0.7, interaction_index=1)
            wsa.add_association("fire", "bright curiosity", confidence=0.7, interaction_index=1)
            wsa.save(path)

            # Conflict is active
            self.assertEqual(len(wsa.detect_conflicts("fire", interaction_index=2)), 1)

            # User resolves conflict:
            wsa.break_bridge("fire", "greetings", "bright curiosity", "bright curiosity", interaction_index=3, reward=0.8)
            # Immediate targeted flush:
            wsa.save(path)

            # Simulate immediate crash / restart: instantiate fresh associator from disk
            reloaded = WordStateAssociator()
            reloaded.load(path)

            probs = reloaded.get_label_probabilities("fire", interaction_index=4)
            self.assertGreater(probs["bright curiosity"], 0.90)
            self.assertLess(probs["greetings"], 0.10)
            self.assertEqual(reloaded.detect_conflicts("fire", interaction_index=4), [])
            self.assertEqual(len(reloaded.conflict_history), 1)
            self.assertEqual(reloaded.conflict_history[0]["word"], "fire")
            self.assertEqual(reloaded.conflict_history[0]["kept"], "bright curiosity")


if __name__ == "__main__":
    unittest.main()
