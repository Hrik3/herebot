import os
import tempfile
import unittest

import numpy as np

import components.factor_handler as factor_handler
from components.factor_handler import StateManager
from components.learner import WordStateAssociator
from concepts.hdp import HierarchicalConceptModel


class TestProbabilisticConceptModel(unittest.TestCase):
    def test_rewarded_concept_probability_promotes(self):
        model = HierarchicalConceptModel(alpha=0.5, beta=1.0)
        model.concept_counts[0] = 1
        model.concept_counts[1] = 1
        model.word_concept_counts["unit"][0] = 1
        model.word_concept_counts["unit"][1] = 1
        model.get_concept_probabilities("unit", interaction_index=1)

        for turn in range(2, 20):
            model.record_concept_reward("unit", 0, interaction_index=turn, reward=2.0, weight=1.0)

        probs = model.get_concept_probabilities("unit", interaction_index=20)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[0], 0.75)
        self.assertLessEqual(probs[0], 1.0)

    def test_unused_concept_decays_by_interaction_gap_and_archives(self):
        model = HierarchicalConceptModel(alpha=0.5, beta=1.0)
        model.concept_counts[0] = 1
        model.word_concept_counts["unit"][0] = 1
        stats = model._ensure_word_concept_stats("unit", interaction_index=0)
        stats[0]["usage"] = 0.01
        stats[0]["reward_trace"] = -5.0
        stats[0]["last_interaction"] = 0
        stats[0]["last_decay_interaction"] = 0

        active = model.get_concept_probabilities("unit", interaction_index=10000)
        archived = model.get_concept_probabilities("unit", interaction_index=10000, include_archived=True)

        self.assertEqual(active, {})
        self.assertIn(0, archived)
        self.assertEqual(model.word_concept_stats["unit"][0]["phase"], "archived")

    def test_negative_reward_increases_archive_pressure(self):
        model = HierarchicalConceptModel(alpha=0.5, beta=1.0)
        model.concept_counts[0] = 1
        model.word_concept_counts["unit"][0] = 1
        model.get_concept_probabilities("unit", interaction_index=1)
        before = model.word_concept_stats["unit"][0]["archive_pressure"]

        model.record_concept_reward("unit", 0, interaction_index=2, reward=-3.0, weight=1.0)
        model.get_concept_probabilities("unit", interaction_index=500)
        after = model.word_concept_stats["unit"][0]["archive_pressure"]

        self.assertGreaterEqual(after, before)


class TestProbabilisticWordStateLabels(unittest.TestCase):
    def test_multiple_labels_coexist_and_reward_promotes_one(self):
        assoc = WordStateAssociator()
        assoc.add_association("unit", "alpha", confidence=1.0, interaction_index=1, reward=0.0)
        assoc.add_association("unit", "beta", confidence=1.0, interaction_index=2, reward=0.0)

        for turn in range(3, 12):
            assoc.add_association("unit", "beta", confidence=1.0, interaction_index=turn, reward=2.0)

        probs = assoc.get_label_probabilities("unit", interaction_index=12)
        self.assertIn("alpha", probs)
        self.assertIn("beta", probs)
        self.assertGreater(probs["beta"], probs["alpha"])
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_break_bridge_demotes_rejected_label(self):
        assoc = WordStateAssociator()
        assoc.add_association("unit", "alpha", confidence=1.0, interaction_index=1)
        assoc.add_association("unit", "beta", confidence=1.0, interaction_index=2)
        before = assoc.get_label_probabilities("unit", interaction_index=2)["beta"]

        assoc.break_bridge("unit", "alpha", "beta", "alpha", interaction_index=3, reward=-2.0)
        after = assoc.get_label_probabilities("unit", interaction_index=3).get("beta", 0.0)

        self.assertLess(after, before)
        self.assertGreater(assoc.get_label_probabilities("unit", interaction_index=3)["alpha"], after)

    def test_archived_label_is_not_returned(self):
        assoc = WordStateAssociator()
        assoc.add_association("unit", "alpha", confidence=1.0, interaction_index=1)
        assoc.word_label_stats["unit"]["alpha"]["phase"] = "archived"
        assoc._normalize_word_labels("unit", interaction_index=2)

        self.assertEqual(assoc.get_labels("unit", threshold=0.0), [])


class TestProbabilisticStateLabels(unittest.TestCase):
    def make_manager(self, tmpdir):
        old_state_dir = factor_handler.STATE_DIR
        old_conflict_file = StateManager.CONFLICT_FILE
        factor_handler.STATE_DIR = tmpdir
        StateManager.CONFLICT_FILE = os.path.join(tmpdir, "conflicts.json")
        manager = StateManager(state_dir=os.path.join(tmpdir, "states"))
        self.addCleanup(setattr, factor_handler, "STATE_DIR", old_state_dir)
        self.addCleanup(setattr, StateManager, "CONFLICT_FILE", old_conflict_file)
        return manager

    def test_multiple_similar_states_can_be_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            manager.states["alpha"] = np.array([1.0, 0.0])
            manager.states["beta"] = np.array([1.0, 0.0])
            manager._ensure_state_probability_metadata("alpha", interaction_index=1)
            manager._ensure_state_probability_metadata("beta", interaction_index=1)
            manager._normalize_state_probabilities(["alpha", "beta"], interaction_index=1)

            matches = manager.find_similar_states(np.array([1.0, 0.0]), threshold=0.99, interaction_index=2)

            self.assertEqual({label for label, _ in matches}, {"alpha", "beta"})

    def test_rewarded_similar_state_is_selected_over_unused_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            manager.states["alpha"] = np.array([1.0, 0.0])
            manager.states["beta"] = np.array([1.0, 0.0])
            manager._ensure_state_probability_metadata("alpha", interaction_index=1)
            manager._ensure_state_probability_metadata("beta", interaction_index=1)
            manager._normalize_state_probabilities(["alpha", "beta"], interaction_index=2, reward=3.0, touched_label="beta", weight=5.0)

            selected = manager.find_similar_state(np.array([1.0, 0.0]), threshold=0.99, interaction_index=3)

            self.assertEqual(selected, "beta")

    def test_exact_tie_uses_random_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            manager.states["alpha"] = np.array([1.0, 0.0])
            manager.states["beta"] = np.array([1.0, 0.0])
            manager.state_metadata["alpha"] = {
                "usage": 1.0, "reward_trace": 0.0, "last_interaction": 1,
                "last_decay_interaction": 1, "probability": 0.5, "tier": 2, "phase": "active"
            }
            manager.state_metadata["beta"] = {
                "usage": 1.0, "reward_trace": 0.0, "last_interaction": 1,
                "last_decay_interaction": 1, "probability": 0.5, "tier": 2, "phase": "active"
            }

            called = {"value": False}
            original_choice = factor_handler.np.random.choice

            def fake_choice(n, *args, **kwargs):
                called["value"] = True
                return 1

            factor_handler.np.random.choice = fake_choice
            self.addCleanup(setattr, factor_handler.np.random, "choice", original_choice)

            selected = manager.find_similar_state(np.array([1.0, 0.0]), threshold=0.99, interaction_index=1)

            self.assertTrue(called["value"])
            self.assertEqual(selected, "beta")

    def test_archive_pressure_uses_interaction_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            manager.states["alpha"] = np.array([1.0, 0.0])
            manager.state_metadata["alpha"] = {
                "usage": 0.01, "reward_trace": -5.0, "last_interaction": 0,
                "last_decay_interaction": 0, "probability": 1.0, "tier": 4, "phase": "active"
            }

            active = manager.get_state_probabilities(interaction_index=10000)
            archived = manager.get_state_probabilities(interaction_index=10000, include_archived=True)

            self.assertEqual(active, {})
            self.assertIn("alpha", archived)
            self.assertEqual(manager.state_metadata["alpha"]["phase"], "archived")


class TestAutomaticStateRegrouping(unittest.TestCase):
    def make_manager(self, tmpdir):
        old_state_dir = factor_handler.STATE_DIR
        old_conflict_file = StateManager.CONFLICT_FILE
        factor_handler.STATE_DIR = tmpdir
        StateManager.CONFLICT_FILE = os.path.join(tmpdir, "conflicts.json")
        manager = StateManager(state_dir=os.path.join(tmpdir, "states"))
        self.addCleanup(setattr, factor_handler, "STATE_DIR", old_state_dir)
        self.addCleanup(setattr, StateManager, "CONFLICT_FILE", old_conflict_file)
        return manager

    def seed_state(self, manager, label, vector, interaction=1, usage=1.0, reward_trace=0.0):
        manager.states[label] = np.array(vector, dtype=float)
        manager.state_metadata[label] = {
            "usage": usage,
            "reward_trace": reward_trace,
            "last_interaction": interaction,
            "last_decay_interaction": interaction,
            "probability": 0.5,
            "tier": 2,
            "phase": "active",
            "tokens": [],
        }

    def test_similar_states_do_not_create_unresolved_interactive_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            self.seed_state(manager, "alpha", [1.0, 0.0])
            self.seed_state(manager, "beta", [1.0, 0.0])

            manager.refresh_conflicts()
            observations = manager.refresh_state_relations(interaction_index=1, tokens=["focus"], reward=1.0)

            self.assertEqual(manager.get_unresolved_conflicts(), [])
            self.assertEqual(len(observations), 1)
            self.assertIn("alpha||beta", manager.state_relation_observations)

    def test_repeated_similar_states_form_head_label_without_losing_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            self.seed_state(manager, "alpha", [1.0, 0.0], reward_trace=1.0)
            self.seed_state(manager, "beta", [1.0, 0.0], reward_trace=1.0)

            for turn in (1, 2, 10):
                manager.record_state_relation_observation(
                    ["alpha", "beta"],
                    tokens=["focus", "build"],
                    reward=1.0,
                    interaction_index=turn,
                )
            manager.refresh_state_regrouping(interaction_index=10, force=True)

            groups = list(manager.state_groups.values())
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["head_label"], "focus build")
            self.assertEqual(set(groups[0]["child_labels"]), {"alpha", "beta"})
            self.assertIn("alpha", manager.states)
            self.assertIn("beta", manager.states)
            self.assertEqual(manager.get_group_heads_for_labels(["alpha", "beta"]), ["focus build"])

    def test_diverged_states_do_not_regroup_until_reconverged_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            self.seed_state(manager, "alpha", [1.0, 0.0])
            self.seed_state(manager, "beta", [0.0, 1.0])

            manager.refresh_state_relations(interaction_index=1, tokens=["split"], reward=1.0)
            manager.refresh_state_regrouping(interaction_index=10, force=True)
            self.assertEqual(manager.state_groups, {})

            manager.states["beta"] = np.array([1.0, 0.0])
            for turn in (11, 12, 20):
                manager.record_state_relation_observation(
                    ["alpha", "beta"],
                    tokens=["merge"],
                    reward=1.0,
                    interaction_index=turn,
                )
            manager.refresh_state_regrouping(interaction_index=20, force=True)

            self.assertEqual(len(manager.state_groups), 1)
            self.assertEqual(list(manager.state_groups.values())[0]["head_label"], "merge")

    def test_group_centroid_uses_probability_and_reward_trace_weighting(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            self.seed_state(manager, "alpha", [1.0, 0.0], usage=10.0, reward_trace=3.0)
            self.seed_state(manager, "beta", [0.0, 1.0], usage=1.0, reward_trace=0.0)

            centroid = manager._compute_group_centroid(["alpha", "beta"], interaction_index=1)

            self.assertGreater(centroid[0], centroid[1])

    def test_archived_states_are_excluded_from_relation_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.make_manager(tmp)
            self.seed_state(manager, "alpha", [1.0, 0.0])
            self.seed_state(manager, "beta", [1.0, 0.0])
            manager.state_metadata["beta"]["phase"] = "archived"

            observations = manager.refresh_state_relations(interaction_index=1, tokens=["focus"], reward=1.0)

            self.assertEqual(observations, [])
            self.assertEqual(manager.state_relation_observations, {})


if __name__ == "__main__":
    unittest.main()
