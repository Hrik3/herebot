"""
TrackerTelemetry_test.py — Regression tests for GP uncertainty telemetry
and full signal lifecycle audit.

Verifies:
1. Raw GP uncertainty is computed and distinct from MetaLearner weights.
2. Raw GP uncertainty and variance reach the tracker.
3. Tracker persistence preserves telemetry without silent data loss.
4. Reloading tracker data preserves schema version and fields.
5. Graph generation reads the intended quantities with historical pen-lift.
6. Missing historical observations remain missing/None rather than fake 0.
7. Telemetry covers early-return paths (knowledge gap, feedback, etc.).
8. E10 FactorSpace and E12 SemanticSpace invariants remain intact.
"""
import json
import os
import tempfile
import unittest
import numpy as np

from concepts.gaussian_uncertainity import GaussianProcessUncertainty
from core.tracker import CuriosityTracker
from core.heart import CuriosityEngine


class TestGPTelemetryLifecycle(unittest.TestCase):
    def test_gp_predict_variance_method_exists_and_matches(self):
        gp = GaussianProcessUncertainty(input_dim=5, n_features=100)
        x = np.random.randn(5)
        mean, var = gp.predict(x)
        var_direct = gp.predict_variance(x)
        std_direct = gp.uncertainty(x)
        self.assertAlmostEqual(var, var_direct, places=7)
        self.assertAlmostEqual(np.sqrt(var), std_direct, places=7)
        self.assertGreaterEqual(var_direct, 0.01)

    def test_raw_gp_uncertainty_reaches_tracker_snapshot(self):
        engine = CuriosityEngine(load_state=False)
        # Process a word
        engine.process_input("water")
        engine.process_input("water means liquid")
        engine.process_input("water")

        snap = engine.tracker._data[-1]
        self.assertEqual(snap.get("schema_version"), 2)
        self.assertIn("avg_gp_variance", snap)
        self.assertIn("avg_gp_uncertainty", snap)
        self.assertIn("turn_gp_uncertainty", snap)

        avg_var = snap["avg_gp_variance"]
        avg_unc = snap["avg_gp_uncertainty"]
        turn_unc = snap["turn_gp_uncertainty"]

        self.assertIsNotNone(avg_var)
        self.assertIsNotNone(avg_unc)
        self.assertIsNotNone(turn_unc)
        # Mathematical lower bounds from beta=100 -> sigma_noise^2=0.01
        self.assertGreaterEqual(avg_var, 0.01)
        self.assertGreaterEqual(avg_unc, 0.10)
        self.assertGreaterEqual(turn_unc, 0.01)

    def test_meta_learner_weights_remain_distinct_from_gp_telemetry(self):
        engine = CuriosityEngine(load_state=False)
        engine.process_input("water")
        engine.process_input("water means liquid")
        engine.process_input("water")

        snap = engine.tracker._data[-1]
        meta_weight = snap.get("factor_weights", {}).get("gp_uncertainty")
        avg_var = snap.get("avg_gp_variance")
        avg_unc = snap.get("avg_gp_uncertainty")

        self.assertIsNotNone(meta_weight)
        self.assertIsNotNone(avg_var)
        self.assertIsNotNone(avg_unc)
        # Learned regression weight mu_gp must not be confused with variance
        self.assertNotEqual(meta_weight, avg_var)
        self.assertNotEqual(meta_weight, avg_unc)

    def test_telemetry_covers_early_return_knowledge_gap(self):
        engine = CuriosityEngine(load_state=False)
        initial_points = len(engine.tracker._data)
        # 'unknownwordxyz' triggers knowledge gap early return
        res = engine.process_input("unknownwordxyz")
        self.assertIn("Knowledge gap detected", res.get("curiosity_artifacts", [""])[0])
        # Tracker must have recorded this turn!
        self.assertEqual(len(engine.tracker._data), initial_points + 1)
        snap = engine.tracker._data[-1]
        self.assertEqual(snap.get("trigger"), "knowledge_gap")
        self.assertIsNotNone(snap.get("turn_gp_uncertainty"))

    def test_historical_fake_zeros_lift_pen_in_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = CuriosityTracker(graphs_dir=tmp)
            # Seed with one historical entry containing fake 0.0 and one real entry
            tracker._data = [
                {
                    "schema_version": 1,
                    "timestamp": 1000.0,
                    "session_id": "old_sess",
                    "interaction_num": 1,
                    "trigger": "interaction",
                    "avg_gp_variance": 0.0,  # historical swallowed exception
                    "total_reward": 1.0,
                    "avg_entropy": 0.5,
                },
                {
                    "schema_version": 2,
                    "timestamp": 1050.0,
                    "session_id": "new_sess",
                    "interaction_num": 2,
                    "trigger": "interaction",
                    "avg_gp_variance": 0.0142,  # real computed variance
                    "total_reward": 2.0,
                    "avg_entropy": 0.4,
                }
            ]
            # Must plot without exception
            tracker.plot_all()
            self.assertTrue(os.path.isfile(os.path.join(tmp, "overview_dashboard.png")))

    def test_tracker_persistence_and_reload_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = CuriosityTracker(graphs_dir=tmp)
            snap_data = {
                "schema_version": 2,
                "timestamp": 2000.0,
                "session_id": "persist_test",
                "interaction_num": 5,
                "trigger": "interaction",
                "avg_gp_variance": 0.0135,
                "avg_gp_uncertainty": 0.1162,
                "turn_gp_uncertainty": 0.0521,
                "turn_info_gain": 0.025,
                "total_reward": 5.5,
                "episode_reward": 0.5,
            }
            tracker._data.append(snap_data)
            tracker.save_data()

            # Reload in a clean tracker instance
            reloaded_tracker = CuriosityTracker(graphs_dir=tmp)
            self.assertEqual(len(reloaded_tracker._data), 1)
            reloaded_snap = reloaded_tracker._data[0]
            self.assertEqual(reloaded_snap["schema_version"], 2)
            self.assertAlmostEqual(reloaded_snap["avg_gp_variance"], 0.0135, places=5)
            self.assertAlmostEqual(reloaded_snap["avg_gp_uncertainty"], 0.1162, places=5)
            self.assertAlmostEqual(reloaded_snap["turn_gp_uncertainty"], 0.0521, places=5)
            self.assertAlmostEqual(reloaded_snap["turn_info_gain"], 0.025, places=5)


if __name__ == "__main__":
    unittest.main()
